#!/bin/bash
# One GGUF on stock llama.cpp b12110 llama-server: IFEval passes ($PASSES) -> server down.
#   run_gguf_model.sh <tag> <gguf path> <port>
# Server: --jinja (the GGUF's embedded LFM3 chat template), --reasoning-format deepseek (thinking -> reasoning_content, like
# vLLM's qwen3 parser), min_p off and repeat penalty 1.0 like vLLM; requests carry temperature / top_p / max_tokens.
# Decode batches on this build run the NVFP4 FF at W4A8; prompt batches at W4A4 for an untagged NVFP4 file (Liquid's)
# and W4A8 for a prec_a4=false tagged one (mine).
# top_k 40 (llama.cpp default), NOT 0: with top_k disabled the CPU sampler sorts the full 128k vocabulary for top_p on
# every token (measured: 3.3x slower). top_k 40 is a no-op for greedy and, at temperature 0.6 / top_p 0.95, the 95%
# nucleus essentially never exceeds 40 tokens, so the sampled distribution matches vLLM's (no top_k) in practice.
# EXTRA="-bs" adds backend (GPU) sampling: greedy output verified identical to CPU sampling (bs_check.py).
# Unified KV (-kvu) is what produced the published runs: one shared pool, so a single long thinker can use most of it.
# Env: LLAMA_BIN, CTX (default 524288), NP (slots, default 48), CONC (lm-eval concurrency, default 48), PASSES
# (default "liquid greedy"), EXTRA (extra llama-server flags).
set -u
ROOT=$(cd "$(dirname "$0")" && pwd)
B=${LLAMA_BIN:-/d/AI/tools/llama.cpp-b12110}
TAG=$1; GGUF=$2; PORT=$3
CTX=${CTX:-524288}; NP=${NP:-48}; PASSES=${PASSES:-"liquid greedy"}; EXTRA=${EXTRA:-}
export CONC=${CONC:-48}
mkdir -p "$ROOT/logs"
echo "== $TAG (gguf) start $(date '+%I:%M:%S %p')"
nohup "$B/llama-server" -m "$GGUF" --alias lfm3 --host 127.0.0.1 --port "$PORT" -c "$CTX" -np "$NP" -kvu -ngl 99 -fa on \
  -t 16 -b 2048 -ub 512 --jinja --reasoning-format deepseek --top-k 40 --min-p 0 --repeat-penalty 1.0 $EXTRA \
  > "$ROOT/logs/llamaserver_${TAG}.log" 2>&1 &
SPID=$!
for i in $(seq 1 60); do curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && break; sleep 2; done
curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' || { echo "llama-server $TAG not ready"; kill $SPID 2>/dev/null; exit 2; }
echo "== $TAG server ready, top_k 40, NP $NP, concurrency $CONC, extra [$EXTRA] $(date '+%I:%M:%S %p')"
for m in $PASSES; do bash "$ROOT/run_ifeval.sh" "$TAG" "$PORT" "$m"; done
kill $SPID 2>/dev/null
echo "== $TAG (gguf) done $(date '+%I:%M:%S %p')"
