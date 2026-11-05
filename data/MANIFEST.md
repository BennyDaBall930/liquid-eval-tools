# LFM3-1.7B NVFP4 eval set: frozen held-out evaluation data

Frozen 2026-11-01. Generated with LiquidAI/LFM3-1.7B @ac2586a3 in BF16 (transformers 5.20.2, torch 2.14.1+cu130, sdpa
attention, RTX 5090). Every sequence points at its public source row (`eval_meta.json`), so the prompts can be re-derived
from the pinned dataset revisions below. The frozen files and their sha256 are the reference, not a rerun (sampling on a
shared GPU is not bit-exact).

## Files (sha256)

| file | bytes | sha256 |
|---|---|---|
| eval.pt | 1,379,711 | `b3192d527b2377f9b2c05e806eb5143d50abfd453c112771b7c16c317c7cdc50` |
| eval_ranges.json | 1,818 | `95dd05cd8eb5c8135f05d74e3631dfc9dd3c0cbaacbbc3963a9b6f6fa6eaa020` |
| eval_text.txt | 699,399 | `6cf0a48f02ed5bd379867a6747e6e62a0163455ebc9bae3e3576457a89e40f7f` |
| eval_meta.json | 61,310 | `517454289805ec752e67e42ab621ca290c582ed3fc30323049b803feabd9256f` |

`eval_records.jsonl` (ids + metadata as plain JSON) and `eval_decoded.jsonl` (decoded text, split into prompt / model /
tool-result segments, from `tools/dump_eval.py`) are derived from these.

## Format
- `eval.pt`: `torch.load` -> Python list of 1-D int64 tensors (token ids), each <= 2048 tokens. Order: tool, chat, reason,
  raw. Chat sequences are exact token ids (prompt rendered by the model's own chat_template.jinja + the BF16 model's
  sampled output ids, no re-tokenization). Raw sequences = `<|startoftext|>` + text tokens.
- `eval_ranges.json`: list of `[start, end)` per sequence (same order): chat = the final assistant response (first generated
  token through `<|im_end|>`; for 2-turn tool samples, the response after the tool result); raw = `[32, len)`.
- `eval_meta.json`: per sequence category, source repo/revision/file/row, thinking flag, system prompt, stage2 flag,
  truncation flag, `gen_ranges` (every model-generated span, incl. the first tool-call turn). `kld.py` scores `gen_ranges`.
- `eval_text.txt`: every sequence decoded with `skip_special_tokens=True` (drops `<|startoftext|>`, `<|im_start|>`,
  `<|im_end|>`, which llama-perplexity would not parse as specials; `<think>`, `<tool_call>`, `<tool_response>`, `<tools>`
  are non-special added tokens and stay), joined by blank lines. Re-tokenized: 167,320 tokens.

## Generation settings
- tool / chat: `{'do_sample': True, 'temperature': 0.1, 'top_k': 50, 'repetition_penalty': 1.1}` (model card 'General');
  reason (math): `{'do_sample': True, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'repetition_penalty': 1.0}` (model
  card 'Math & STEM').
- max_new_tokens: {'tool': 1024, 'chat': 1536, 'reason': 1536}, total sequence cap 2048, prompt cap 1024 tokens (longer
  prompts skipped).
- thinking: template default is ON (`<think>` opened by the generation prompt). tool + chat use enable_thinking=False for
  ~25% of prompts; reason always ON.
- system prompts: tool [None, 'You are a helpful assistant trained by Liquid AI.', 'You are a helpful assistant with access
  to tools. Call a tool when it helps answer the user.']; chat [None, None, None, 'You are a helpful assistant trained by
  Liquid AI.'] (uniform choice). Dataset system prompts (Hermes / ToolACE format instructions) are dropped; tool schemas go
  through the template's native `tools=` argument.
- tool 2-turn: if the model's own first turn emits `<tool_call>`s whose function names all match the dataset's recorded
  tool results (Hermes func-calling, ToolACE first call), those results are appended as `tool` turns exactly as the
  template renders them and the model generates its follow-up (same sampling). Otherwise single turn.
- batched HF `generate`, left padding with attention mask, per-batch seed.
- WildChat filter: toxic=False, redacted=False, LFM3 languages only, first user turn 16-6000 chars.

## Sources (pinned revisions)
| repo | revision | file |
|---|---|---|
| NousResearch/hermes-function-calling-v1 | dae3e1d28cfbcf4b915c04ea1e072030529b4bda | func-calling.json |
| Team-ACE/ToolACE | 6bda777c88d21e5a204703c1ee45597a8fa4f734 | data.json |
| allenai/WildChat-1M | 7d6490e462285cf85d91eabea0f9a954fbddcd1f | data/train-00000-of-00014.parquet |
| openai/gsm8k | 740312add88f781978c0658806c59bc2815b9866 | main/train-00000-of-00001.parquet |
| AI-MO/NuminaMath-CoT | 9d8d210c9f6a36c8f3cd84045668c9b7800ef517 | data/train-00000-of-00005.parquet |
| Salesforce/wikitext | b08601e04326c79dfdd32d625aee71d232d685c3 | wikitext-103-raw-v1/test-00000-of-00001.parquet |
| HuggingFaceFW/fineweb-edu | 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 | sample/10BT/000_00000.parquet |
| teknium/OpenHermes-2.5 | b82037821055c377bed0d495e72e46de3bc72e84 | openhermes2_5.json |

Row indices for every sequence are in `eval_meta.json` (`row`; for wikitext the first line index of the article in the
parquet).

## Counts

150 sequences, 168,242 tokens, max len 2048

| category | seqs | tokens | share | scored tokens (final response) | scored tokens (all generated spans) |
|---|---|---|---|---|---|
| tool | 57 | 58,890 | 35.0% | 20,089 | 26,631 |
| chat | 48 | 59,453 | 35.3% | 50,389 | 50,389 |
| reason | 18 | 17,870 | 10.6% | 16,605 | 16,605 |
| raw | 27 | 32,029 | 19.0% | 31,165 | 31,165 |

| category | source | seqs | tokens |
|---|---|---|---|
| chat | openhermes:CamelAI | 3 | 3,491 |
| chat | openhermes:EvolInstruct_70k | 2 | 2,730 |
| chat | openhermes:airoboros2.2 | 2 | 934 |
| chat | openhermes:glaive-code-assist | 3 | 3,971 |
| chat | openhermes:metamath | 4 | 1,966 |
| chat | openhermes:none | 6 | 5,133 |
| chat | openhermes:platypus | 1 | 482 |
| chat | wildchat | 27 | 40,746 |
| raw | fineweb-edu | 17 | 14,041 |
| raw | wikitext-103-test | 10 | 17,988 |
| reason | gsm8k | 9 | 8,123 |
| reason | numina:cn_k12 | 4 | 4,254 |
| reason | numina:orca_math | 2 | 2,694 |
| reason | numina:synthetic_amc | 1 | 1,725 |
| reason | numina:synthetic_math | 2 | 1,074 |
| tool | hermes-fc | 23 | 32,784 |
| tool | toolace | 34 | 26,106 |

Flags: chat/think_off=10, chat/think_on=38, chat/truncated=26, reason/think_on=18, reason/truncated=6,
tool/stage2_tool_result_turn=22, tool/think_off=17, tool/think_on=40, tool/truncated=10

WildChat languages: English 14, Russian 3, Chinese 3, Portuguese 3, German 1, Vietnamese 1, Italian 1, French 1

## Observations about LFM3's template (from building the set)

- Thinking is ON by default in the template (the generation prompt ends in `<think>`). With thinking on, chat traces are
  long at temp 0.1 and about half hit the 1536-token cap. The model reasons in English even when the prompt is Italian,
  French, Russian and so on, then answers in the prompt language.
- With `enable_thinking=False` (prompt ends in `<think></think>`), the model sometimes writes its reasoning as plain content
  and closes it with a stray `</think>` before the `<tool_call>`. Liquid's card already warns that the no-think toggle is
  inconsistent. These are kept as-is because they are the model's own outputs.
- Tool calls use the XML form `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`.
  Parallel calls are separated by a blank line. Tool schemas are rendered as one JSON object per line inside
  `<tools>...</tools>`, appended to the system prompt. Each tool result is its own `<|im_start|>tool\n<tool_response>\n...
  \n</tool_response><|im_end|>` turn. `<think>`, `</think>`, `<tool_call>`, `<tool_response>` and `<tools>` are
  NON-special added tokens (single ids 124901/124902/125019/125025/125017). Only `<|startoftext|>`, `<|im_start|>`,
  `<|im_end|>` and `<|pad|>` are special.
- vocab_size in config is 128000, but special/added token ids reach 125026. All ids are < 128000.
