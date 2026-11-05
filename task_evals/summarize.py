"""IFEval results for the five LFM3-1.7B checkpoints -> RESULTS.md + results.json (reads ifeval/<tag>_<mode>/*/results_*.json
and samples_ifeval_*.jsonl written by run_ifeval.sh). Also paired per-prompt comparisons (McNemar, exact two-sided) and
the no-answer count (responses with no final answer: the thinking trace never closed before the model stopped).
Usage: python summarize.py
"""
import glob, json, math, os

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = [  # tag, label, runtime
    ("bf16", "BF16 (LiquidAI/LFM3-1.7B @ac2586a3)", "vLLM 0.32.0 + PR 64959"),
    ("gguf_bdb", "My NVFP4 GGUF (BennyDaBall/LFM3-1.7B-NVFP4-GGUF @dbf6d349)", "llama.cpp b12110"),
    ("gguf_liquid", "Liquid NVFP4 GGUF (LiquidAI/LFM3-1.7B-GGUF @cd0c74c5)", "llama.cpp b12110"),
    ("liquid", "Liquid NVFP4 (LiquidAI/LFM3-1.7B-NVFP4 @bc0c77de)", "vLLM 0.32.0 + PR 64959, W4A4"),
    ("bdb", "My NVFP4 (BennyDaBall/LFM3-1.7B-NVFP4 @32b8d12b)", "vLLM 0.32.0 + PR 64959, W4A4"),
]
MODES = [("liquid", "temperature 0.6, top_p 0.95 (Liquid's model-card protocol)"), ("greedy", "greedy (temperature 0)")]
PAIRS = [("gguf_bdb", "gguf_liquid", "GGUF pair: mine vs Liquid's, llama.cpp"), ("bdb", "liquid", "safetensors pair: mine vs Liquid's, vLLM W4A4"),
         ("gguf_bdb", "bf16", "my GGUF vs BF16"), ("gguf_liquid", "bf16", "Liquid's GGUF vs BF16"),
         ("bdb", "bf16", "my NVFP4 vs BF16 (vLLM)"), ("liquid", "bf16", "Liquid's NVFP4 vs BF16 (vLLM)")]


def run_dir(tag, mode):
    return os.path.join(ROOT, "ifeval", f"{tag}_{mode}")


def latest(pattern):
    f = sorted(glob.glob(pattern), key=os.path.getmtime)
    return f[-1] if f else None


def load(tag, mode):
    r = latest(os.path.join(run_dir(tag, mode), "*", "results_*.json"))
    s = latest(os.path.join(run_dir(tag, mode), "*", "samples_ifeval*.jsonl"))
    if not r or not s:
        return None
    res = json.load(open(r, encoding="utf-8"))
    m = {k.replace(",none", ""): v for k, v in res["results"]["ifeval"].items() if k.endswith(",none")}
    rows = [json.loads(l) for l in open(s, encoding="utf-8")]
    resp = [x["resps"][0][0] for x in rows]
    m["n"] = len(rows)
    m["no_answer"] = sum(1 for t in resp if t == "" or "LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER" in t)
    m["per_prompt_strict"] = {x["doc_id"]: bool(x["prompt_level_strict_acc"]) for x in rows}
    m["gen_kwargs"] = res["config"].get("gen_kwargs")
    m["files"] = [os.path.relpath(r, ROOT).replace("\\", "/"), os.path.relpath(s, ROOT).replace("\\", "/")]
    return m


def mcnemar_p(b, c):
    """exact two-sided binomial test on the discordant pairs"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


D = {f"{t}_{m}": load(t, m) for t, _, _ in MODELS for m, _ in MODES}
L = ["# LFM3-1.7B task evals: IFEval", "",
     "One RTX 5090, 2026-11-05. lm-evaluation-harness 0.4.13 (task `ifeval` v4, 541 prompts), chat template applied, thinking ON "
     "(template default), no output cap below the 262k context (max_tokens 250,000), 48 concurrent requests. The servers' reasoning "
     "parsers return the thinking trace separately, so only the final answer is scored. Prompt-level strict accuracy is the headline; "
     "its standard error is about 0.9 points at 541 prompts.", ""]
for mode, desc in MODES:
    L += [f"## {desc}", "", "| checkpoint | runtime | prompt strict | inst strict | prompt loose | inst loose | no answer |", "|---|---|---|---|---|---|---|"]
    for tag, label, rt in MODELS:
        r = D[f"{tag}_{mode}"]
        if not r:
            L.append(f"| {label} | {rt} | not run | | | | |")
            continue
        L.append(f"| {label} | {rt} | {100 * r['prompt_level_strict_acc']:.2f} ± {100 * r['prompt_level_strict_acc_stderr']:.2f} | "
                 f"{100 * r['inst_level_strict_acc']:.2f} | {100 * r['prompt_level_loose_acc']:.2f} | {100 * r['inst_level_loose_acc']:.2f} | {r['no_answer']} |")
    L += ["", "Paired by prompt (prompt-level strict): prompts that pass on A only vs on B only, exact McNemar p.", "",
          "| A vs B | A only | B only | p |", "|---|---|---|---|"]
    for a, b, name in PAIRS:
        ra, rb = D[f"{a}_{mode}"], D[f"{b}_{mode}"]
        if not ra or not rb:
            continue
        pa, pb = ra["per_prompt_strict"], rb["per_prompt_strict"]
        x = sum(1 for k in pa if pa[k] and not pb.get(k)); y = sum(1 for k in pa if pb.get(k) and not pa[k])
        L.append(f"| {name} | {x} | {y} | {mcnemar_p(x, y):.2f} |")
    L.append("")
L += ["## Reading it", "",
      "- Every checkpoint is within about one standard error of every other under both protocols, and no paired comparison is significant. "
      "At 1.7B, IFEval does not resolve the KLD differences between these NVFP4 files (W4A4 teacher KLD 0.045 mine vs 0.063 Liquid's on vLLM; "
      "llama.cpp 0.043 vs 0.143 as shipped).",
      "- Orderings flip between runtimes and protocols (my vLLM file is lowest at temperature 0.6, my GGUF ties BF16), which is what noise looks like. "
      "At temperature 0.6 each run draws one sample per prompt; at greedy, thinking traces of hundreds of tokens amplify tiny numeric differences, "
      "so quantized and BF16 trajectories diverge within a few dozen tokens either way.",
      "- 'no answer' = the response had no final answer because the trace never closed; under greedy this happens on a handful of prompts for every "
      "checkpoint, on different prompts for each.", "",
      "## Run notes", "",
      "- vLLM: real W4A4 kernels (FlashInferCutlassNvFp4LinearKernel) for both NVFP4 files, 1.86 GiB weights each; BF16 3.36 GiB.",
      "- llama.cpp: top_k 40 (llama.cpp default) on both GGUFs; vLLM applies no top_k. top_k 40 cannot change greedy output and at temperature 0.6 / "
      "top_p 0.95 the nucleus essentially never exceeds 40 tokens. Liquid's GGUF at temperature 0.6 used CPU sampling with 32 slots; its greedy pass and "
      "both of my GGUF passes used backend (GPU) sampling, verified to give identical greedy output (bs_check.json), with 32 slots (Liquid's greedy) or "
      "48 slots (mine). Slot count and sampler location change speed, not the sampled distribution.",
      "- An earlier BF16 pair of runs with a 16,384-token output cap was discarded: the cap cut off 7 long thinkers at temperature 0.6 and made BF16 "
      "look 1.8 points worse. All published runs use the context-bound cap.", ""]
open(os.path.join(ROOT, "RESULTS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L))
out = {k: ({kk: vv for kk, vv in v.items() if kk != "per_prompt_strict"} if v else None) for k, v in D.items()}
json.dump(out, open(os.path.join(ROOT, "results.json"), "w", encoding="utf-8"), indent=1)
print("\n".join(L))
