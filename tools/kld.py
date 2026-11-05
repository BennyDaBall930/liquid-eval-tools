"""Teacher-forced KLD(BF16 || variant) on the frozen eval set, all variants in one process against one BF16 teacher pass.

Gold = LiquidAI/LFM3-1.7B BF16 in transformers (same model object, FF linears unpatched). Each variant = a ModelOpt
NVFP4 safetensors file, fake-quantized W4A4 exactly like the kernels (lfm3q.act_quant) or W4A16 with --a16.
Usage: python kld.py --var liquid=C:\\...\\liquid-nvfp4\\model.safetensors --var ours=...\\model.safetensors [--a16] [--limit N]
"""
import argparse, json, math, os, time
import torch
import torch.nn.functional as F

import lfm3q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root: data/ and results/ live here

ap = argparse.ArgumentParser()
ap.add_argument("--var", action="append", required=True, help="name=path to ModelOpt NVFP4 model.safetensors")
ap.add_argument("--eval", default=os.path.join(ROOT, "data", "eval.pt"))
ap.add_argument("--ranges", default=os.path.join(ROOT, "data", "eval_meta.json"),
                help="eval_meta.json: score every model-generated span (gen_ranges, incl. tool-call turns), per category")
ap.add_argument("--a16", action="store_true", help="weight-only (activations BF16) instead of W4A4")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

torch.manual_seed(0)
seqs = torch.load(args.eval)
if os.path.exists(args.ranges):
    ranges = [{"spans": m.get("gen_ranges") or [m["score"]], "category": m["category"]} for m in json.load(open(args.ranges))]
else:
    ranges = [{"spans": [[32, len(s)]]} for s in seqs]
if args.limit:
    seqs, ranges = seqs[:args.limit], ranges[:args.limit]
model = lfm3q.load_bf16()
variants = {}
for v in args.var:
    name, path = v.split("=", 1)
    variants[name] = lfm3q.load_modelopt_ff(path)
kl_all = {n: [] for n in variants}
top1 = {n: 0 for n in variants}
cat_kl = {n: {} for n in variants}
ntok = 0
t0 = time.time()
with torch.inference_mode():
    for si, (ids, r) in enumerate(zip(seqs, ranges)):
        ids = ids.view(1, -1).cuda()
        # logits at position t predict token t+1: score tokens a..b-1 of each span -> logits a-1..b-2
        sl = torch.cat([torch.arange(max(a - 1, 0), b - 1) for a, b in r["spans"]]).cuda()
        lp_ref = F.log_softmax(model(ids).logits[0, sl].float(), -1)
        am_ref = lp_ref.argmax(-1)
        ntok += lp_ref.shape[0]
        for n, qff in variants.items():
            orig = lfm3q.patch_ff(model, qff, a4=not args.a16)
            lp = F.log_softmax(model(ids).logits[0, sl].float(), -1)
            lfm3q.unpatch_ff(model, orig)
            kl = (lp_ref.exp() * (lp_ref - lp)).sum(-1)
            kl_all[n].append(kl.cpu())
            top1[n] += int((lp.argmax(-1) == am_ref).sum())
            c = r.get("category", "all")
            cat_kl[n].setdefault(c, []).append(kl.cpu())
        if si % 20 == 0:
            print(f"{si}/{len(seqs)} tok {ntok} {time.time() - t0:.0f}s " +
                  " ".join(f"{n}={torch.cat(kl_all[n]).mean():.5f}" for n in variants), flush=True)
res = {"gold": "LiquidAI/LFM3-1.7B BF16 transformers forward (same process)", "mode": "W4A16" if args.a16 else "W4A4",
       "tokens": ntok, "sequences": len(seqs), "variants": {}}
for n in variants:
    k = torch.cat(kl_all[n])
    res["variants"][n] = {"kld_mean": k.mean().item(), "kld_median": k.median().item(),
                          "kld_p99": k.quantile(0.99).item() if k.numel() < 16_000_000 else None,
                          "kld_max": k.max().item(), "top1_agree": top1[n] / ntok,
                          "by_category": {c: torch.cat(v).mean().item() for c, v in cat_kl[n].items()}}
    print(n, json.dumps(res["variants"][n]))
json.dump(res, open(args.out, "w"), indent=1)
