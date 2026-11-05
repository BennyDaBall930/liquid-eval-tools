# LFM3-1.7B GGUF: KLD vs the BF16 GGUF on llama.cpp

Same eval text for every row: `data/eval_text.txt` (sha256 `6cf0a48f...`, 167,320 tokens, 81 chunks x 2048 ctx, whole
file), `llama-perplexity -c 2048 -b 2048 -ub 512 -ngl 99 -fa on --kl-divergence`, base = the official
`LFM3-1.7B-BF16.gguf` logits. Script `tools/gguf_kld.sh`; rows in `results/gguf_kld.jsonl`. RTX 5090.

## How llama.cpp runs an NVFP4 GGUF (matters for the two NVFP4 rows)

- Decode batches (<= 8 tokens) are always W4A8 (`vec_dot_nvfp4_q8_1`, Q8_1 activations), so token-generation speed is the
  same either way.
- Prompt / batched processing on Blackwell defaults to W4A4 for an NVFP4 file that carries no precision tag: per-token fp32
  activation scale, per-16 E4M3 block scale, E2M1 RNE, native FP4 MMA. A file tagged `general.tensor_extra.prec_a4 = false`
  (or the env `GGML_CUDA_MMQ_PREC=q8`) runs W4A8 there too. KLD runs go through this batched path, so an untagged file is
  measured at W4A4 and a tagged one at W4A8. Liquid's NVFP4 GGUF is untagged; mine is tagged.
- The ModelOpt static `input_scale` in the file is loaded but not used by llama.cpp.

## 2026-11-01, llama.cpp b12015 / b12032 (CUDA 13.2): official files

BF16 base PPL 4.4737 on the scored halves. At the time, stock llama.cpp did not apply the NVFP4 per-tensor scale for the
LFM feed-forward (fixed upstream in b12088), so the Liquid NVFP4 rows here come from a build with that one fix.

| GGUF | activations | size GB | KLD mean | KLD median | KLD p99 | top-1 agree % | PPL |
|---|---|---|---|---|---|---|---|
| BF16 (official) | - | 3.467 | 0 (base) | - | - | 100 | 4.4737 |
| Q8_0 (official) | Q8_1 | 1.846 | 0.00280 | 0.00044 | 0.0366 | 98.24 | 4.4991 |
| Q4_K_M (official) | Q8_1 | 1.100 | 0.10102 | 0.02327 | 1.4907 | 89.22 | 4.5372 |
| QAD-Q4_0 (official) | Q8_1 | 1.050 | 0.10522 | 0.01883 | 1.5912 | 89.33 | 4.4469 |
| Liquid NVFP4 safetensors converted, W4A4 (default) | NVFP4 dyn. | 1.839 | 0.14527 | 0.03229 | 2.2593 | 87.25 | 4.5926 |
| Liquid NVFP4 safetensors converted, W4A8 (`GGML_CUDA_MMQ_PREC=q8`) | Q8_1 | 1.839 | 0.08851 | 0.01811 | 1.4155 | 90.06 | 4.5428 |

Cross-checks: Q4_K_M on release b12015 gives the identical KLD row (0.101019 / 89.223%); BF16 on b12015 vs the b12032
base = KLD 0.00012, top-1 99.60% (build-to-build noise floor).

## 2026-11-04, stock llama.cpp b12110 (v0.9.0, CUDA 13.4), nothing patched: the two public NVFP4 GGUFs

Liquid published `LFM3-1.7B-NVFP4.gguf` in LiquidAI/LFM3-1.7B-GGUF on 2026-11-03 (revision `cd0c74c5`, sha256
`462c2379...`): converted from the same ModelOpt checkpoint as their safetensors, NVFP4 feed-forward + Q8_0 for everything
else, untagged, 1.19 GiB. Mine (BennyDaBall/LFM3-1.7B-NVFP4-GGUF, revision `dbf6d349`, sha256 `df905d8a...`) has the same
construction and size, with the `prec_a4 = false` tag. New BF16 base written with the same build (PPL 4.4931). Speed from
`llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 5 -t 16`, GPU otherwise idle.

| GGUF (both 1.19 GiB) | precision llama.cpp uses | KLD mean | KLD median | KLD p99 | top-1 agree % | PPL | tg128 tok/s | pp512 tok/s |
|---|---|---|---|---|---|---|---|---|
| Liquid NVFP4 GGUF, as shipped (untagged: W4A4 prompt path) | NVFP4 dyn. | 0.14274 | 0.03191 | 2.1598 | 87.43 | 4.6296 | 597 +- 4 | 49,462 +- 5,012 |
| Liquid NVFP4 GGUF, `GGML_CUDA_MMQ_PREC=q8` | Q8_1 | 0.09014 | 0.01854 | 1.4068 | 89.96 | 4.5436 | same file | same file |
| Mine, as shipped (tag pins W4A8) | Q8_1 | 0.04357 | 0.00963 | 0.6047 | 92.54 | 4.5689 | 598 +- 15 | 39,912 +- 2,080 |
| Q8_0 (official) | Q8_1 | 0.00277 | | | 98.23 | | 497 | |
| Q4_K_M (official) | Q8_1 | 0.09694 | | | 89.33 | | 634 | |

- Like for like at W4A8: 0.0901 vs 0.0436 = 2.07x. As users get the two files on stock llama.cpp: 0.1427 vs 0.0436 =
  3.28x, because Liquid's file is untagged and Blackwell's batched path takes NVFP4 weights to W4A4 for prompt processing.
  Decode is W4A8 for both regardless of the tag, so tg128 is identical. Their untagged file is about 24% faster on pp512
  (FP4 tensor-core MMQ) at the cost of the 0.14 KLD.
- The current Hub revision of my GGUF (`dbf6d349`, 2026-11-04) measures 0.04278 on this same setup.
- Both NVFP4 files are Q8_0-sized and decode at Q8_0-class speed; as shipped, Liquid's sits below Q4_K_M in quality (0.143
  vs 0.097) at Q4_K_M-class speed, and only beats Q4_K_M when forced to W4A8.
