#!/bin/bash
# IFEval (lm-evaluation-harness 0.4.13, task ifeval v4, 541 prompts) against a running OpenAI-compatible endpoint
# (vLLM or llama-server) that serves the model as "lfm3".
#   run_ifeval.sh <tag> <port> <liquid|greedy>
# liquid = Liquid's model-card protocol for its general evals: temperature 0.6, top_p 0.95, no output cap below the
#          262k context (max_tokens 250,000; IFEval prompts are short). greedy = temperature 0, same cap.
# Thinking stays ON (template default). The server's reasoning parser (vLLM --reasoning-parser qwen3, llama-server
# --reasoning-format deepseek) returns the trace as reasoning_content, so the scored `content` is the answer only.
# Env: PY (python with lm_eval), CONC (concurrent requests, default 48), LIMIT (smoke run on the first N prompts).
set -u
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
TAG=$1; PORT=$2; MODE=$3
case "$MODE" in
  liquid) GK="temperature=0.6,top_p=0.95,max_gen_toks=250000" ;;
  greedy) GK="temperature=0,max_gen_toks=250000" ;;
  *) echo "mode must be liquid or greedy"; exit 1 ;;
esac
OUT=$ROOT/ifeval/${TAG}_${MODE}${LIMIT:+_limit$LIMIT}
LOG=$ROOT/logs/ifeval_${TAG}_${MODE}${LIMIT:+_limit$LIMIT}.log
mkdir -p "$OUT" "$ROOT/logs"
t0=$(date +%s)
"$PY" -m lm_eval --model local-chat-completions \
  --model_args "model=lfm3,base_url=http://127.0.0.1:${PORT}/v1/chat/completions,num_concurrent=${CONC:-48},max_retries=5,tokenized_requests=False" \
  --tasks ifeval --apply_chat_template --gen_kwargs "$GK" --seed 1234 --batch_size 1 \
  --output_path "$OUT" --log_samples ${LIMIT:+--limit $LIMIT} > "$LOG" 2>&1
rc=$?
echo "ifeval $TAG $MODE rc=$rc $(( $(date +%s) - t0 )) s"
