"""Merge kld_exp.py shard outputs (<out>.kl.pt files) into one kld.py-format JSON with exact full-set metrics.
Usage: python merge_kld.py --out merged.json shard0.json.kl.pt shard1.json.kl.pt ..."""
import argparse, json
import torch

ap = argparse.ArgumentParser()
ap.add_argument("shards", nargs="+")
ap.add_argument("--out", required=True)
args = ap.parse_args()
parts = [torch.load(p) for p in args.shards]
names = list(parts[0]["variants"])
ntok = sum(p["tokens"] for p in parts)
res = {"gold": "LiquidAI/LFM3-1.7B BF16 transformers forward (same process per shard)", "mode": parts[0]["mode"],
       "tokens": ntok, "sequences": sum(p["sequences"] for p in parts), "shards": [p["shard"] for p in parts], "variants": {}}
for n in names:
    k = torch.cat([p["variants"][n]["kl"] for p in parts])
    assert k.numel() == ntok
    top1 = sum(p["variants"][n]["top1"] for p in parts)
    cats = {}
    for p in parts:
        for c, v in p["variants"][n]["cat_kl"].items():
            cats.setdefault(c, []).append(v)
    res["variants"][n] = {"kld_mean": k.mean().item(), "kld_median": k.median().item(),
                          "kld_p99": k.quantile(0.99).item(), "kld_max": k.max().item(), "top1_agree": top1 / ntok,
                          "by_category": {c: torch.cat(v).mean().item() for c, v in cats.items()}}
    print(n, json.dumps(res["variants"][n]))
json.dump(res, open(args.out, "w"), indent=1)
print("tokens", ntok, "->", args.out)
