#!/bin/bash
# One safetensors checkpoint on vLLM (real W4A4 NVFP4 kernels for the quants): server -> IFEval (Liquid protocol, then
# greedy) -> server down.
#   run_model.sh <tag> <model dir name under $LFM3_MODELS> <host port>
# Server: vLLM v0.32.0 + vllm-project/vllm PR 64959 (LFM3 support, head 5f67fbcb, vllm/ part only) as a Docker image
# ($VLLM_IMAGE), full 262k context, reasoning + tool parsers, --generation-config vllm (no sampling defaults from
# generation_config.json, so temperature 0 is greedy and repetition penalty 1.0), default output cap 32,768 tokens for
# clients that send no max_tokens. lm-eval sends its own max_tokens.
# Env: LFM3_MODELS (host folder with the checkpoint dirs), VLLM_IMAGE, CONC (IFEval concurrency, default 48).
set -u
ROOT=$(cd "$(dirname "$0")" && pwd)
MODELS=${LFM3_MODELS:-C:/AI/Models/lfm3}
IMG=${VLLM_IMAGE:-lfm3-vllm:v0.32.0-pr64959}
TAG=$1; DIR=$2; PORT=$3
mkdir -p "$ROOT/logs"
echo "== $TAG start $(date '+%I:%M:%S %p')"
MSYS_NO_PATHCONV=1 docker run -d --rm --name "lfm3-$TAG" --gpus all --ipc=host -p "$PORT:8000" \
  -v "$MODELS:/models:ro" -v lfm3-vllm-cache:/root/.cache "$IMG" \
  --model "/models/$DIR" --served-model-name lfm3 --trust-remote-code \
  --gpu-memory-utilization 0.35 --max-model-len 262144 --max-num-seqs 64 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --generation-config vllm --override-generation-config '{"max_new_tokens": 32768}' > /dev/null
nohup docker logs -f "lfm3-$TAG" > "$ROOT/logs/vllm_$TAG.log" 2>&1 &
for i in $(seq 1 90); do curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" > /dev/null && break; sleep 5; done
curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "server $TAG not ready after 450 s"; exit 2; }
echo "== $TAG server ready $(date '+%I:%M:%S %p')"
grep -E "Using .*Kernel for NVFP4|Model loading took" "$ROOT/logs/vllm_$TAG.log" | tail -2
bash "$ROOT/run_ifeval.sh" "$TAG" "$PORT" liquid
bash "$ROOT/run_ifeval.sh" "$TAG" "$PORT" greedy
docker stop "lfm3-$TAG" > /dev/null 2>&1
echo "== $TAG done $(date '+%I:%M:%S %p')"
