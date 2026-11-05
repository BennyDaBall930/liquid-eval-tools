"""(1) Greedy decode on held-out eval prompts with BF16, Liquid NVFP4 and mine (W4A4 fake quant), counting how many tokens
each quant stays identical to BF16; (2) teacher-forced per-token KLD + top-1 strips for two eval samples.
Output: results/greedy_agree.json.
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F

import lfm3q

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--new", type=int, default=256)
ap.add_argument("--liquid", default=os.path.join(lfm3q.LIQUID_DIR, "model.safetensors"),
                help="LiquidAI/LFM3-1.7B-NVFP4 model.safetensors")
ap.add_argument("--ours", required=True, help="the ModelOpt NVFP4 model.safetensors to compare (BennyDaBall/LFM3-1.7B-NVFP4)")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "greedy_agree.json"))
args = ap.parse_args()

seqs = torch.load(os.path.join(ROOT, "data", "eval.pt"))
meta = json.load(open(os.path.join(ROOT, "data", "eval_meta.json"), encoding="utf-8"))
tok = lfm3q.load_tokenizer()
model = lfm3q.load_bf16()
V = {"liquid": lfm3q.load_modelopt_ff(args.liquid), "ours": lfm3q.load_modelopt_ff(args.ours)}

# chat-style prompts only (tool / chat / reason), spread across categories, deterministic order
idx = [i for i, m in enumerate(meta) if m["category"] != "raw"]
pick = idx[::max(1, len(idx) // args.n)][:args.n]


def greedy(ids):
    out = model.generate(ids, max_new_tokens=args.new, do_sample=False, repetition_penalty=1.0)
    return out[0, ids.shape[1]:].tolist()


res = {"greedy": [], "strips": []}
t0 = time.time()
with torch.inference_mode():
    for k, i in enumerate(pick):
        p = meta[i]["n_prompt"]
        ids = seqs[i][:p].view(1, -1).cuda()
        ref = greedy(ids)
        row = {"eval_idx": i, "category": meta[i]["category"], "key": meta[i]["key"], "bf16": ref}
        for n, q in V.items():
            orig = lfm3q.patch_ff(model, q)
            g = greedy(ids)
            lfm3q.unpatch_ff(model, orig)
            same = next((j for j, (a, b) in enumerate(zip(ref, g)) if a != b), min(len(ref), len(g)))
            row[n] = g
            row[n + "_agree"] = same
        res["greedy"].append(row)
        print(f"{k}/{len(pick)} {meta[i]['category']} liquid={row['liquid_agree']} ours={row['ours_agree']} "
              f"{time.time() - t0:.0f}s", flush=True)
    # per-token strips: first tool sample with a 2-turn answer and the first chat sample
    for cat in ("tool", "chat"):
        i = next(j for j, m in enumerate(meta) if m["category"] == cat and m.get("gen_ranges"))
        a, b = meta[i]["gen_ranges"][-1]
        ids = seqs[i].view(1, -1).cuda()
        sl = torch.arange(a - 1, b - 1).cuda()
        lp_ref = F.log_softmax(model(ids).logits[0, sl].float(), -1)
        strip = {"eval_idx": i, "category": cat, "tokens": [tok.decode([t]) for t in seqs[i][a:b].tolist()]}
        for n, q in V.items():
            orig = lfm3q.patch_ff(model, q)
            lp = F.log_softmax(model(ids).logits[0, sl].float(), -1)
            lfm3q.unpatch_ff(model, orig)
            strip[n + "_kld"] = (lp_ref.exp() * (lp_ref - lp)).sum(-1).tolist()
            strip[n + "_top1"] = (lp.argmax(-1) == lp_ref.argmax(-1)).tolist()
        res["strips"].append(strip)
ag = {n: [r[n + "_agree"] for r in res["greedy"]] for n in V}
res["summary"] = {n: {"mean_tokens_identical": sum(v) / len(v), "median": sorted(v)[len(v) // 2],
                      "full_match_256": sum(x >= args.new for x in v), "n": len(v)} for n, v in ag.items()}
res["settings"] = {"new_tokens": args.new, "greedy": True, "mode": "W4A4 fake quant (kernel-exact math)"}
print(json.dumps(res["summary"]))
json.dump(res, open(args.out, "w"))
