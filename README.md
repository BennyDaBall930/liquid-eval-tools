# LFM3-1.7B NVFP4 eval tools

The harness, frozen data and raw results behind the KLD and BF16-agreement numbers on my two LFM3-1.7B NVFP4 repos,
[BennyDaBall/LFM3-1.7B-NVFP4](https://huggingface.co/BennyDaBall/LFM3-1.7B-NVFP4) (ModelOpt safetensors, vLLM / SGLang)
and [BennyDaBall/LFM3-1.7B-NVFP4-GGUF](https://huggingface.co/BennyDaBall/LFM3-1.7B-NVFP4-GGUF) (llama.cpp).

What this is, honestly: a one-person, one-GPU (RTX 5090) experiment in pushing teacher KLD and weight error lower for
NVFP4 at the same file size and speed, by replacing round-to-nearest with my own calibrated solver. I am an enthusiast,
not a lab. The main eval here is a
teacher-forced, distribution-level measurement against the BF16 model, not a task benchmark. On 2026-11-05 I added IFEval
on all five checkpoints with Liquid's card settings (see [Task evals](#task-evals-ifeval-2026-11-05)). BFCL V4 is not
included: its agentic web-search category needs a paid search API key, and without it BFCL's overall score cannot be
compared with published numbers, so this repo sticks to what it can show in full: the real-world tool-call slice (Hermes
and ToolACE prompts, below) and IFEval as a baseline task comparison.
Everything needed to recompute every number is here or pinned by revision.

## Layout

```
tools/
  dump_eval.py         decodes data/eval.pt into readable JSONL (prompt / model spans / tool-result turns)
  kld.py               teacher-forced KLD(BF16 || variant), W4A4 or W4A16, all variants in one process
  kld_exp.py           same metric, memory-light (packed tensors on CPU, dequant in forward), sharded
  merge_kld.py         merges kld_exp.py shards into exact full-set metrics
  paired_boot.py       paired per-token bootstrap over sequences, variants vs a reference variant
  greedy_agree.py      greedy continuation agreement vs BF16 + per-token KLD strips
  smoke_chat_tools.py  one chat + one tool-call request against an OpenAI endpoint (vLLM / SGLang / llama-server)
  gguf_kld.sh          llama-perplexity --kl-divergence on data/eval_text.txt against the BF16 GGUF
  lfm3q.py             NVFP4 fake quant for the FF path (ModelOpt layout, kernel-matching activation quant)
  nvfp4q.py            E2M1 / E4M3 grid primitives (encode, decode, dequant)
data/
  MANIFEST.md          how the sets were built, settings, sources, counts, sha256 of every file, observations
  eval.pt              150 held-out sequences, token ids (list of int64 tensors)      sha256 b3192d52...
  eval_meta.json       per sequence: category, source repo/revision/file/row, flags, scored spans
  eval_ranges.json     final-response span per sequence (the GGUF path scores the whole text instead)
  eval_text.txt        eval sequences decoded without special tokens, for llama-perplexity    sha256 6cf0a48f...
  eval_records.jsonl   the same 150 sequences as plain JSON (ids + metadata), no torch needed
  eval_decoded.jsonl   the same 150 sequences decoded to text, split into prompt / model / tool-result segments
  sources.json         pinned source datasets (repo, revision, file)
results/
  kld_w4a4.json        2026-11-01: Liquid NVFP4 vs my first build, W4A4 (the numbers the first release shipped with)
  kld_w4a16.json       2026-11-01: same, weight-only
  greedy_agree.json    2026-11-01: greedy continuation agreement + per-token strips (Liquid vs my first build)
  gguf_kld.md/.jsonl   llama.cpp path: GGUF files vs the BF16 GGUF (Q8_0, Q4_K_M, QAD-Q4_0, both NVFP4 GGUFs)
  smoke_*.json         raw vLLM request/response for one chat and one tool-call prompt per file
task_evals/
  RESULTS.md           IFEval for BF16 + four NVFP4 files, both protocols, paired per-prompt tests, run notes
  results.json         the same numbers as JSON, with the lm-eval file each one came from
  ifeval/<tag>_<mode>/ lm-eval results_*.json and samples_ifeval_*.jsonl (every prompt, response and per-instruction verdict)
  run_ifeval.sh        lm-eval IFEval against an OpenAI-compatible endpoint (both protocols)
  run_model.sh         vLLM (Docker, PR 64959 overlay) server + IFEval
  run_gguf_model.sh    llama.cpp llama-server + IFEval
  bs_check.py / bs_check.json   llama.cpp CPU vs GPU sampling equivalence check (greedy, identical output)
  summarize.py         builds RESULTS.md / results.json from the lm-eval outputs
```

## The eval set

Built once on 2026-11-01 and frozen; full detail, generation settings, pinned sources and all sha256 values in
[data/MANIFEST.md](data/MANIFEST.md). Four categories, 150 sequences, 168,242 tokens, 124,790 of them scored:

| category | sequences | scored tokens | what is scored |
|---|---|---|---|
| tool | 57 | 26,631 | every model-generated span: thinking, tool calls, and the turn after the tool result |
| chat | 48 | 50,389 | the model's response (thinking + answer) |
| reason | 18 | 16,605 | the model's response, math prompts at the card's math sampling |
| raw | 27 | 31,165 | tokens 32..end of wikitext-103 test articles and fineweb-edu documents |

Chat, tool and reason prompts are real public prompts rendered with LFM3's own `chat_template.jinja`; the responses are
the BF16 LFM3-1.7B's own generations (HF `generate`, sampling from the model card: temperature 0.1, top_k 50,
repetition_penalty 1.1; math: temperature 1.0, top_p 0.95, top_k 20), stored as the exact generated token ids, never
re-tokenized. So the KLD is measured on-policy: on text the BF16 model itself would write.

### The tool-call slice

This is the part I was asked about, so here it is in full.

- 57 sequences: 23 rows from `NousResearch/hermes-function-calling-v1` (`func-calling.json`, revision `dae3e1d2`) and
  34 rows from `Team-ACE/ToolACE` (`data.json`, revision `6bda777c`). Row indices per sequence are in
  `data/eval_meta.json` (`row`), the decoded text is in `data/eval_decoded.jsonl`.
- Prompt construction: only the dataset's first user turn is used. The dataset's own system prompt (Hermes / ToolACE
  format instructions) is dropped. The tool schemas are passed through the template's native `tools=` argument, so LFM3
  renders them its own way (one JSON object per line inside `<tools>...</tools>` in the system turn). ToolACE schemas
  with `"type": "dict"` were normalized to `"object"`. The system prompt is one of three, chosen uniformly:
  none (16 sequences), `You are a helpful assistant trained by Liquid AI.` (22), and
  `You are a helpful assistant with access to tools. Call a tool when it helps answer the user.` (19).
- Thinking: the template default (on) for 40 sequences; `enable_thinking=False` for 17.
- Generation: BF16 model, batched HF `generate` with left padding, temperature 0.1, top_k 50, repetition_penalty 1.1,
  max_new_tokens 1024, seeded per batch. 10 of the 57 hit the length cap.
- Second turn: when the model's first turn emitted `<tool_call>` blocks whose function names all matched the dataset's
  recorded tool results, those results were appended as `<|im_start|>tool` turns exactly as the template renders them
  and the model generated its follow-up with the same sampling. 22 of the 57 sequences have this second turn.
- Scored tokens: 26,631 (20,741 in the first turn, which includes the thinking and the tool-call XML; 5,890 in the
  post-tool-result turns). Prompts are 101 to 978 tokens (median 495). Scored spans are `gen_ranges` in
  `data/eval_meta.json`; `kld.py` scores every span in that list.
- Format facts that matter for anyone scoring tool calls: LFM3 writes calls as
  `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`; `<think>`, `</think>`,
  `<tool_call>`, `<tool_response>` and `<tools>` are single non-special added tokens (ids 124901 / 124902 / 125019 /
  125025 / 125017); only `<|startoftext|>`, `<|im_start|>`, `<|im_end|>` and `<|pad|>` are special.

Caveats I would want to know if I were reading this:

- The prompts come from two public SFT datasets. LFM3 may have seen them or close relatives in training. That does not
  change what the metric measures (how closely the quant tracks the BF16 model on its own outputs), but it is not a
  held-out task set.
- Low-temperature self-generated trajectories are the right distribution for teacher forcing a quant against its own
  BF16 parent, and the wrong thing to read as "tool-calling accuracy". A quant that tracks the BF16 distribution more
  closely should score closer to BF16 on tasks, in either direction; whether that shows up above benchmark noise at this
  model size is exactly what I have not measured.
- 57 sequences is a modest sample. Between my own builds the tool-category
  mean moved by up to 6% while the full-set mean moved by under 2%, so read the tool number as a direction, not a
  precise ranking, and expect nothing useful about tails from it.

## The metrics, exactly

**Teacher-forced KLD** (`tools/kld.py`, `tools/kld_exp.py`). Gold is `LiquidAI/LFM3-1.7B` BF16 (revision `ac2586a3`,
weights sha256 `135922ca...`) in transformers 5.20.2 with sdpa attention, run in the same process as every variant. For
each scored token position, logits are upcast to float32, `log_softmax` over the full vocabulary, and the per-token
value is `KL(P_bf16 || P_variant) = sum_v p_bf16(v) * (log p_bf16(v) - log p_variant(v))`. Reported: mean over scored
tokens (the headline number), median, p99, max, top-1 agreement (the variant's argmax equals BF16's argmax at that
position), and the mean per category. `kld_exp.py` computes the identical quantity with the packed tensors streamed to
the GPU per variant; `merge_kld.py` concatenates shards so sharded runs give the exact full-set values.

**What a variant is.** A ModelOpt NVFP4 `model.safetensors` (Liquid's layer map: the 72 feed-forward linears `w1`,
`w3`, `w2` of 24 layers are NVFP4; attention, short convolutions, embeddings, head and norms stay BF16 and are
byte-identical to the base checkpoint in both Liquid's file and mine). `lfm3q.load_modelopt_ff` dequantizes each FF
weight to float32 as `e2m1(code) * e4m3(block_scale) * weight_scale_2` and `lfm3q.patch_ff` swaps the model's FF
`nn.Linear` modules for `FQLinear`:

- **W4A4** (default): the input is fake-quantized along the last dim in blocks of 16 with the file's static per-tensor
  `input_scale`: block scale `sf = e4m3_rne(min(bmax / 6 / input_scale, 448))`, values `e2m1_rne(x / (sf * input_scale))`
  saturating at 6, then dequantized; the GEMM is `F.linear` in float32 on the dequantized operands. This mirrors the
  CUTLASS / FlashInfer block-scaled FP4 GEMM (exact FP4 x FP4 products, fp32 accumulation) up to summation order. I have
  not diffed it against vLLM's kernel logits token for token; the equivalence is by construction, and the vLLM smoke
  tests in `results/smoke_*.json` are generation-level sanity checks, not a numerical proof.
- **W4A16** (`--a16`): weights dequantized as above, activations left in BF16. This isolates the weight error.

**Paired bootstrap** (`tools/paired_boot.py`). For each variant, the per-token KL difference against a reference variant
is summed per sequence; 1,000 resamples draw 150 sequences with replacement (seed 0), and each resample's statistic is the
summed difference divided by the summed token count (so it is the token-weighted mean delta). CI95 is the 2.5 / 97.5
percentile. This is how I decide whether a new build of mine replaces the previous one.

**Greedy agreement** (`tools/greedy_agree.py`). 40 held-out eval prompts (19 tool, 16 chat, 5 reason), greedy decoding
for 256 new tokens with BF16, Liquid's NVFP4 and my file (W4A4 fake quant), counting identical tokens before the first
divergence from the BF16 continuation. It is a harsh metric (one different token ends the count) at small n; I report
it per category below because the categories disagree.

**GGUF path** (`tools/gguf_kld.sh`). `llama-perplexity --kl-divergence` with the official `LFM3-1.7B-BF16.gguf` as base
on `data/eval_text.txt` (the eval set decoded without special tokens, 167,320 tokens, 81 chunks of 2048; the whole
text is scored, not only the model spans). Note llama.cpp's own NVFP4 behaviour: decode batches (<= 8 tokens) are
always W4A8; prompt batches on Blackwell run an untagged NVFP4 file at W4A4 with llama.cpp's own dynamic per-token
activation scale, and a file tagged `prec_a4 = false` (or `GGML_CUDA_MMQ_PREC=q8`) at W4A8. The ModelOpt static
`input_scale` is loaded but not used by llama.cpp. Details and table in [results/gguf_kld.md](results/gguf_kld.md).

## Results in this repo

Checkpoints, pinned:

| file | Hub revision | LFS sha256 |
|---|---|---|
| LiquidAI/LFM3-1.7B BF16 (gold) | `ac2586a3` (current `0dc4268b`, weights unchanged) | `135922ca...` |
| LiquidAI/LFM3-1.7B-NVFP4 (RTN, static scales) | `bc0c77de` (weights unchanged since 2026-10-27) | `386de2dd...` |
| BennyDaBall/LFM3-1.7B-NVFP4, first build (2026-11-01, `sr_hq` below) | `9a14adcb` | superseded, fetch by revision |
| BennyDaBall/LFM3-1.7B-NVFP4, current (2026-11-04) | `32b8d12b` | `af9c824e...` |
| BennyDaBall/LFM3-1.7B-NVFP4-GGUF, current (2026-11-04) | `dbf6d349` | `df905d8a...` |
| LiquidAI/LFM3-1.7B-GGUF `LFM3-1.7B-NVFP4.gguf` (2026-11-03) | `cd0c74c5` | `462c2379...` |

W4A4 teacher KLD, 124,790 scored tokens (`results/kld_w4a4.json` for the 11-01 rows; the current-weights row is the
number on the Hub card, same harness):

| variant | mean | median | p99 | top-1 agree | tool | chat | reason | raw |
|---|---|---|---|---|---|---|---|---|
| Liquid NVFP4 | 0.06283 | 0.01872 | 0.5839 | 90.34% | 0.03525 | 0.07069 | 0.03784 | 0.08700 |
| my first build `sr_hq` (11-01) | 0.04612 | 0.01331 | 0.4410 | 91.63% | 0.02326 | 0.05228 | 0.02618 | 0.06631 |
| my current Hub weights (11-04) | 0.04511 | 0.01305 | 0.4256 | 91.83% | 0.02313 | 0.05041 | 0.02543 | 0.06581 |

W4A16 (weights only) teacher KLD (`results/kld_w4a16.json`):

| variant | mean | top-1 agree | tool | chat | reason | raw |
|---|---|---|---|---|---|---|
| Liquid NVFP4 | 0.03317 | 92.93% | 0.01842 | 0.03763 | 0.02075 | 0.04517 |
| `sr_hq` (11-01) | 0.01781 | 94.80% | 0.00774 | 0.02078 | 0.00964 | 0.02598 |
| my current Hub weights (11-04) | 0.01721 | 94.96% | 0.00727 | 0.01995 | 0.00927 | 0.02550 |

Greedy continuation agreement, 256 tokens, mean (median) identical tokens before first divergence from BF16
(`results/greedy_agree.json`, measured 2026-11-01 against my first build, not the current weights):

| subset | n | Liquid NVFP4 | my first build |
|---|---|---|---|
| tool | 19 | 18.5 (9) | 34.8 (13) |
| chat | 16 | 28.3 (23) | 19.9 (23) |
| reason | 5 | 18.6 (22) | 29.2 (22) |
| all | 40 | 22.4 (10) | 28.2 (14) |

The chat subset favours Liquid's file on this metric and the medians are close everywhere; at n = 40 I would not read
more than "both diverge from BF16 within a few dozen greedy tokens, mine a little later on tool prompts" into it. Neither
file produced a full 256-token match.

GGUF (llama.cpp b12110, stock, W4A8 for both unless noted; `results/gguf_kld.md`): Liquid's NVFP4 GGUF 0.1427 as shipped
(untagged, W4A4 prompt path) / 0.0901 at W4A8; mine 0.0428 (tag pins W4A8); official Q8_0 0.0028, Q4_K_M 0.0969.

## Reproducing

```bash
pip install -r requirements.txt     # the pinned versions are what produced the numbers; newer should work
hf download LiquidAI/LFM3-1.7B       --revision ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7 --local-dir models/lfm3-bf16
hf download LiquidAI/LFM3-1.7B-NVFP4 --local-dir models/liquid-nvfp4
hf download BennyDaBall/LFM3-1.7B-NVFP4 --local-dir models/bdb-nvfp4
export LFM3_BF16_DIR=models/lfm3-bf16 LFM3_LIQUID_DIR=models/liquid-nvfp4   # defaults point at my machine's folders
cd tools
python kld.py --var liquid=../models/liquid-nvfp4/model.safetensors --var bdb=../models/bdb-nvfp4/model.safetensors --out ../results/my_w4a4.json
python kld.py --a16 --var liquid=... --var bdb=... --out ../results/my_w4a16.json
python dump_eval.py --category tool --out ../data/tool_slice.jsonl          # prompts as text, CPU only
python greedy_agree.py --ours ../models/bdb-nvfp4/model.safetensors --out ../results/my_greedy.json
```

`kld.py` keeps every variant's dequantized FF weights on the GPU (about 4.5 GB each, plus the 3.4 GB model); with many
variants use `kld_exp.py --shard i/n` in parallel processes, then `merge_kld.py` and `paired_boot.py` on the
`*.kl.pt` shard files. Runtime noise: the BF16 teacher logits move at the 1e-4 KLD level between torch / cuBLAS builds
(measured on the GGUF path: BF16 vs BF16 across two llama.cpp builds = 0.00012), so compare variants inside one run rather
than against the numbers in this repo.

The eval set is frozen; `data/eval_meta.json` carries the source repo, revision, file and row of every prompt, so the
prompts can be re-derived from the pinned public datasets, and the sha256 values in `data/MANIFEST.md` identify the exact
token files these numbers were measured on.

## If you run task evals with matched settings

What I would hold fixed so the comparison is between the FF weights and nothing else:

- Both NVFP4 files have the same layer map, the same format, the same size and (within a few percent, flipping sign
  between runtimes) the same speed. The differences are the 72 FF tensors' E2M1 codes and E4M3 block scales, and my
  static `input_scale` values. Everything else is Liquid's BF16.
- Run the real W4A4 kernels: vLLM 0.32.0 + `vllm-project/vllm` PR 64959 (or SGLang nightly + `sgl-project/sglang`
  PR 46858), as on Liquid's card. For tool calling in vLLM I used `--reasoning-parser qwen3 --tool-call-parser
  qwen3_coder --enable-auto-tool-choice` (`results/smoke_bdb_vllm.json` has the raw request and response).
- Thinking on (the template default) unless the benchmark's protocol says otherwise; same `max_tokens`; either greedy or
  the card's sampling (temperature 0.1, top_k 50, repetition_penalty 1.1) with the same seed for both files.
- My file at revision `32b8d12b`, Liquid's at `bc0c77de`. The first-build numbers in `kld_w4a4.json` belong to
  revision `9a14adcb`, which is still fetchable by revision.

My honest expectation going in was that 0.063 vs 0.045 mean KLD is a real but modest gap on a 1.7B model and that task
deltas would land inside run-to-run noise. IFEval bore that out (see below). The per-category split above says where a
heavier benchmark would have to look first: chat and raw text carry most of the KLD, tool calls the least.

## Task evals: IFEval (2026-11-05)

lm-evaluation-harness 0.4.13, task `ifeval` (541 prompts), chat template applied, thinking on, no output cap below the
262k context, 48 concurrent requests, one RTX 5090. The vLLM rows run the real W4A4 NVFP4 kernels; the GGUF rows run on
stock llama.cpp b12110 (Liquid's file untagged, so W4A4 prompt processing; mine tagged W4A8). Full tables, the greedy
protocol, paired per-prompt tests and run notes: [task_evals/RESULTS.md](task_evals/RESULTS.md); every prompt and
response: `task_evals/ifeval/`.

| temperature 0.6, top_p 0.95 (Liquid's card protocol) | prompt strict | inst strict | prompt loose | inst loose |
|---|---|---|---|---|
| BF16, vLLM | 95.56 ± 0.89 | 97.12 | 96.86 | 97.96 |
| My NVFP4 GGUF, llama.cpp | 95.56 ± 0.89 | 97.00 | 97.23 | 98.20 |
| Liquid NVFP4 GGUF, llama.cpp | 95.19 ± 0.92 | 96.76 | 96.86 | 97.96 |
| Liquid NVFP4, vLLM | 95.01 ± 0.94 | 96.76 | 97.04 | 98.08 |
| My NVFP4, vLLM | 94.45 ± 0.98 | 96.28 | 96.67 | 97.84 |

Every checkpoint is within about one standard error of every other, under this protocol and under greedy, and no paired
comparison is significant (smallest exact McNemar p = 0.38 across twelve pairs). Both of my files keep BF16's instruction
following at the same size and speed; IFEval at 1.7B does not resolve the KLD gap between the NVFP4 files.

## Licenses and provenance

Code in `tools/` and `task_evals/` is MIT (see `LICENSE`). The frozen token sets contain short excerpts (one first user turn, or one article / document, per row)
from the public datasets listed in `data/sources.json`, each under its own license (NousResearch/hermes-function-calling-v1,
Team-ACE/ToolACE, allenai/WildChat-1M, teknium/OpenHermes-2.5, openai/gsm8k, AI-MO/NuminaMath-CoT, Salesforce/wikitext,
HuggingFaceFW/fineweb-edu), plus responses generated by LiquidAI/LFM3-1.7B (LFM Open License v1.0). Not affiliated with
or endorsed by Liquid AI.
