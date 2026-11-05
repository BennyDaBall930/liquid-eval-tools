"""Paired per-token KL comparison of variants against a reference variant, bootstrapped over eval SEQUENCES.
Reads kld_exp.py shard outputs (<out>.kl.pt, which carry per-sequence token counts in 'seq_lens').
Usage: python paired_boot.py --ref liquid --out boot.json shard0.json.kl.pt shard1.json.kl.pt ..."""
import argparse, json
import torch

ap = argparse.ArgumentParser()
ap.add_argument("shards", nargs="+")
ap.add_argument("--ref", required=True, help="name of the reference variant")
ap.add_argument("--resamples", type=int, default=1000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
parts = [torch.load(p) for p in args.shards]
names = list(parts[0]["variants"])
seq_lens = sum((p["seq_lens"] for p in parts), [])
kl = {n: torch.cat([p["variants"][n]["kl"] for p in parts]).double() for n in names}
ntok = sum(seq_lens)
assert all(k.numel() == ntok for k in kl.values())
bounds = torch.tensor([0] + seq_lens).cumsum(0)
nseq = len(seq_lens)
lens = torch.tensor(seq_lens, dtype=torch.float64)
g = torch.Generator().manual_seed(0)
samples = torch.randint(0, nseq, (args.resamples, nseq), generator=g)
res = {"ref": args.ref, "tokens": ntok, "sequences": nseq, "resamples": args.resamples, "arms": {}}
ref = kl[args.ref]
for n in names:
    k = kl[n]
    diff_seq = torch.stack([(k[a:b] - ref[a:b]).sum() for a, b in zip(bounds[:-1], bounds[1:])])   # per-sequence summed diff
    boot = (diff_seq[samples].sum(1) / lens[samples].sum(1))
    lo, hi = boot.quantile(0.025).item(), boot.quantile(0.975).item()
    res["arms"][n] = {"kld_mean": k.mean().item(), "kld_p99": k.quantile(0.99).item(),
                      "delta_mean_vs_ref": (k.mean() - ref.mean()).item(), "delta_ci95": [lo, hi],
                      "delta_rel_pct": 100 * (k.mean() - ref.mean()).item() / ref.mean().item(),
                      "p99_delta_vs_ref": (k.quantile(0.99) - ref.quantile(0.99)).item(),
                      "ci_excludes_zero": bool(hi < 0 or lo > 0)}
    r = res["arms"][n]
    print(f"{n}: mean {r['kld_mean']:.5f} ({r['delta_rel_pct']:+.2f}%) CI95 [{lo:+.5f}, {hi:+.5f}] "
          f"p99 {r['kld_p99']:.4f} ({r['p99_delta_vs_ref']:+.4f}) excl0={r['ci_excludes_zero']}")
json.dump(res, open(args.out, "w"), indent=1)
