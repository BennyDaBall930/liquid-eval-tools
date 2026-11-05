"""kld.py's teacher-forced KLD(BF16 || variant) eval for many variants under a small GPU budget.

Same gold (LFM3-1.7B BF16 transformers, FF linears unpatched, same process), same frozen eval set and spans
(eval.pt + eval_meta.json gen_ranges), same W4A4 fake quant (lfm3q.act_quant on the input, fp32 F.linear on the
dequantized weight) and the same metrics as kld.py. Difference: kld.py holds every variant's dequantized fp32 FF weights
on the GPU (4.5 GB per variant); here each variant's PACKED ModelOpt tensors (0.64 GB) live on the CPU, are moved to the
GPU for that variant's forward only, and each FF linear dequantizes its own weight with lfm3q.dequant inside forward
(the identical fp32 tensor kld.py would have stored). GPU peak ~ model 3.4 GB + 0.64 GB + activations.
Usage: python kld_exp.py --var liquid=<path>/model.safetensors --var bdb=<path>/model.safetensors --out results.json [--a16] [--limit N]
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

TOOLS = os.path.dirname(os.path.abspath(__file__))  # lfm3q.py sits beside this script
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import lfm3q  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--var", action="append", required=True, help="name=path to ModelOpt NVFP4 safetensors (FF keys needed)")
ap.add_argument("--eval", default=os.path.join(ROOT, "data", "eval.pt"))
ap.add_argument("--ranges", default=os.path.join(ROOT, "data", "eval_meta.json"))
ap.add_argument("--a16", action="store_true", help="weight-only (activations BF16) instead of W4A4")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", default="0/1", help="i/n: evaluate sequences i::n (merge shards with merge_kld.py)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
shard_i, shard_n = (int(x) for x in args.shard.split("/"))


class FQLinearPacked(torch.nn.Module):
    """kld.py's lfm3q.FQLinear with the weight dequantized from the packed ModelOpt tensors inside forward."""
    def __init__(self, packed, sf, gs, input_scale):
        super().__init__()
        self.packed, self.sf, self.gs, self.input_scale = packed, sf, gs, input_scale

    def forward(self, x):
        W = lfm3q.dequant(self.packed, self.sf, self.gs)
        xq = lfm3q.act_quant(x, self.input_scale) if self.input_scale is not None else x.float()
        return F.linear(xq, W).to(x.dtype)


def load_packed(path):
    out = {}
    from safetensors import safe_open
    with safe_open(path, "pt", device="cpu") as f:
        for p in lfm3q.ff_names():
            out[p] = dict(packed=f.get_tensor(p + ".weight"), sf=f.get_tensor(p + ".weight_scale"),
                          gs=f.get_tensor(p + ".weight_scale_2"), input_scale=float(f.get_tensor(p + ".input_scale").float()))
    return out


def patch(model, qff, a4):
    orig = {}
    for p, d in qff.items():
        parent = model.get_submodule(p.rsplit(".", 1)[0]); name = p.rsplit(".", 1)[1]
        orig[p] = getattr(parent, name)
        setattr(parent, name, FQLinearPacked(d["packed"].cuda(non_blocking=True), d["sf"].cuda(non_blocking=True),
                                             d["gs"].cuda(), d["input_scale"] if a4 else None))
    return orig


torch.manual_seed(0)
seqs = torch.load(args.eval)
if os.path.exists(args.ranges):
    ranges = [{"spans": m.get("gen_ranges") or [m["score"]], "category": m["category"]} for m in json.load(open(args.ranges))]
else:
    ranges = [{"spans": [[32, len(s)]]} for s in seqs]
if args.limit:
    seqs, ranges = seqs[:args.limit], ranges[:args.limit]
seqs, ranges = seqs[shard_i::shard_n], ranges[shard_i::shard_n]
print(f"shard {args.shard}: {len(seqs)} sequences, {sum(len(s) for s in seqs)} tokens", flush=True)
model = lfm3q.load_bf16()
variants = {}
for v in args.var:
    name, path = v.split("=", 1)
    variants[name] = load_packed(path)
    print(f"loaded {name} ({len(variants[name])} FF linears)", flush=True)
# FQLinearPacked dequantizes with lfm3q.dequant, the same function kld.py's load_modelopt_ff stores the result of, so the
# equivalence was confirmed end to end by reproducing results/kld_w4a4.json.
kl_all = {n: [] for n in variants}
top1 = {n: 0 for n in variants}
cat_kl = {n: {} for n in variants}
ntok = 0
seq_lens = []
t0 = time.time()
with torch.inference_mode():
    for si, (ids, r) in enumerate(zip(seqs, ranges)):
        ids = ids.view(1, -1).cuda()
        sl = torch.cat([torch.arange(max(a - 1, 0), b - 1) for a, b in r["spans"]]).cuda()
        lp_ref = F.log_softmax(model(ids).logits[0, sl].float(), -1)
        am_ref = lp_ref.argmax(-1)
        ntok += lp_ref.shape[0]
        seq_lens.append(int(lp_ref.shape[0]))
        for n, qff in variants.items():
            orig = patch(model, qff, a4=not args.a16)
            lp = F.log_softmax(model(ids).logits[0, sl].float(), -1)
            lfm3q.unpatch_ff(model, orig)
            kl = (lp_ref.exp() * (lp_ref - lp)).sum(-1)
            kl_all[n].append(kl.cpu())
            top1[n] += int((lp.argmax(-1) == am_ref).sum())
            c = r.get("category", "all")
            cat_kl[n].setdefault(c, []).append(kl.cpu())
            del lp
        if si % 20 == 0:
            print(f"{si}/{len(seqs)} tok {ntok} {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB " +
                  " ".join(f"{n}={torch.cat(kl_all[n]).mean():.5f}" for n in variants), flush=True)
res = {"gold": "LiquidAI/LFM3-1.7B BF16 transformers forward (same process)", "mode": "W4A16" if args.a16 else "W4A4",
       "tokens": ntok, "sequences": len(seqs), "shard": args.shard, "seconds": time.time() - t0,
       "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30, "variants": {}}
# raw per-token KLs (+ top1 counts, per-category token KLs) so shards can be merged into the exact full-set metrics
torch.save({"variants": {n: {"kl": torch.cat(kl_all[n]), "top1": top1[n],
                             "cat_kl": {c: torch.cat(v) for c, v in cat_kl[n].items()}} for n in variants},
            "tokens": ntok, "sequences": len(seqs), "shard": args.shard, "mode": res["mode"], "seq_lens": seq_lens},
           args.out + ".kl.pt")
for n in variants:
    k = torch.cat(kl_all[n])
    res["variants"][n] = {"kld_mean": k.mean().item(), "kld_median": k.median().item(),
                          "kld_p99": k.quantile(0.99).item() if k.numel() < 16_000_000 else None,
                          "kld_max": k.max().item(), "top1_agree": top1[n] / ntok,
                          "by_category": {c: torch.cat(v).mean().item() for c, v in cat_kl[n].items()}}
    print(n, json.dumps(res["variants"][n]), flush=True)
json.dump(res, open(args.out, "w"), indent=1)
