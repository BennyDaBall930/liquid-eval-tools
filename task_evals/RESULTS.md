# LFM3-1.7B task evals: IFEval

One RTX 5090, 2026-11-05. lm-evaluation-harness 0.4.13 (task `ifeval` v4, 541 prompts), chat template applied, thinking ON (template default), no output cap below the 262k context (max_tokens 250,000), 48 concurrent requests. The servers' reasoning parsers return the thinking trace separately, so only the final answer is scored. Prompt-level strict accuracy is the headline; its standard error is about 0.9 points at 541 prompts.

## temperature 0.6, top_p 0.95 (Liquid's model-card protocol)

| checkpoint | runtime | prompt strict | inst strict | prompt loose | inst loose | no answer |
|---|---|---|---|---|---|---|
| BF16 (LiquidAI/LFM3-1.7B @ac2586a3) | vLLM 0.32.0 + PR 64959 | 95.56 ± 0.89 | 97.12 | 96.86 | 97.96 | 0 |
| My NVFP4 GGUF (BennyDaBall/LFM3-1.7B-NVFP4-GGUF @dbf6d349) | llama.cpp b12110 | 95.56 ± 0.89 | 97.00 | 97.23 | 98.20 | 0 |
| Liquid NVFP4 GGUF (LiquidAI/LFM3-1.7B-GGUF @cd0c74c5) | llama.cpp b12110 | 95.19 ± 0.92 | 96.76 | 96.86 | 97.96 | 0 |
| Liquid NVFP4 (LiquidAI/LFM3-1.7B-NVFP4 @bc0c77de) | vLLM 0.32.0 + PR 64959, W4A4 | 95.01 ± 0.94 | 96.76 | 97.04 | 98.08 | 0 |
| My NVFP4 (BennyDaBall/LFM3-1.7B-NVFP4 @32b8d12b) | vLLM 0.32.0 + PR 64959, W4A4 | 94.45 ± 0.98 | 96.28 | 96.67 | 97.84 | 0 |

Paired by prompt (prompt-level strict): prompts that pass on A only vs on B only, exact McNemar p.

| A vs B | A only | B only | p |
|---|---|---|---|
| GGUF pair: mine vs Liquid's, llama.cpp | 13 | 11 | 0.84 |
| safetensors pair: mine vs Liquid's, vLLM W4A4 | 10 | 13 | 0.68 |
| my GGUF vs BF16 | 9 | 9 | 1.00 |
| Liquid's GGUF vs BF16 | 10 | 12 | 0.83 |
| my NVFP4 vs BF16 (vLLM) | 13 | 19 | 0.38 |
| Liquid's NVFP4 vs BF16 (vLLM) | 8 | 11 | 0.65 |

## greedy (temperature 0)

| checkpoint | runtime | prompt strict | inst strict | prompt loose | inst loose | no answer |
|---|---|---|---|---|---|---|
| BF16 (LiquidAI/LFM3-1.7B @ac2586a3) | vLLM 0.32.0 + PR 64959 | 94.09 ± 1.02 | 96.04 | 96.30 | 97.48 | 3 |
| My NVFP4 GGUF (BennyDaBall/LFM3-1.7B-NVFP4-GGUF @dbf6d349) | llama.cpp b12110 | 94.64 ± 0.97 | 96.40 | 96.67 | 97.84 | 0 |
| Liquid NVFP4 GGUF (LiquidAI/LFM3-1.7B-GGUF @cd0c74c5) | llama.cpp b12110 | 94.64 ± 0.97 | 96.52 | 96.86 | 97.96 | 0 |
| Liquid NVFP4 (LiquidAI/LFM3-1.7B-NVFP4 @bc0c77de) | vLLM 0.32.0 + PR 64959, W4A4 | 93.90 ± 1.03 | 95.56 | 96.49 | 97.24 | 5 |
| My NVFP4 (BennyDaBall/LFM3-1.7B-NVFP4 @32b8d12b) | vLLM 0.32.0 + PR 64959, W4A4 | 94.09 ± 1.02 | 95.44 | 95.75 | 96.64 | 5 |

Paired by prompt (prompt-level strict): prompts that pass on A only vs on B only, exact McNemar p.

| A vs B | A only | B only | p |
|---|---|---|---|
| GGUF pair: mine vs Liquid's, llama.cpp | 11 | 11 | 1.00 |
| safetensors pair: mine vs Liquid's, vLLM W4A4 | 18 | 17 | 1.00 |
| my GGUF vs BF16 | 13 | 10 | 0.68 |
| Liquid's GGUF vs BF16 | 14 | 11 | 0.69 |
| my NVFP4 vs BF16 (vLLM) | 15 | 15 | 1.00 |
| Liquid's NVFP4 vs BF16 (vLLM) | 12 | 13 | 1.00 |

## Reading it

- Every checkpoint is within about one standard error of every other under both protocols, and no paired comparison is significant. At 1.7B, IFEval does not resolve the KLD differences between these NVFP4 files (W4A4 teacher KLD 0.045 mine vs 0.063 Liquid's on vLLM; llama.cpp 0.043 vs 0.143 as shipped).
- Orderings flip between runtimes and protocols (my vLLM file is lowest at temperature 0.6, my GGUF ties BF16), which is what noise looks like. At temperature 0.6 each run draws one sample per prompt; at greedy, thinking traces of hundreds of tokens amplify tiny numeric differences, so quantized and BF16 trajectories diverge within a few dozen tokens either way.
- 'no answer' = the response had no final answer because the trace never closed; under greedy this happens on a handful of prompts for every checkpoint, on different prompts for each.

## Run notes

- vLLM: real W4A4 kernels (FlashInferCutlassNvFp4LinearKernel) for both NVFP4 files, 1.86 GiB weights each; BF16 3.36 GiB.
- llama.cpp: top_k 40 (llama.cpp default) on both GGUFs; vLLM applies no top_k. top_k 40 cannot change greedy output and at temperature 0.6 / top_p 0.95 the nucleus essentially never exceeds 40 tokens. Liquid's GGUF at temperature 0.6 used CPU sampling with 32 slots; its greedy pass and both of my GGUF passes used backend (GPU) sampling, verified to give identical greedy output (bs_check.json), with 32 slots (Liquid's greedy) or 48 slots (mine). Slot count and sampler location change speed, not the sampled distribution.
- An earlier BF16 pair of runs with a 16,384-token output cap was discarded: the cap cut off 7 long thinkers at temperature 0.6 and made BF16 look 1.8 points worse. All published runs use the context-bound cap.
