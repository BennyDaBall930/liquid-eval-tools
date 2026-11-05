#!/bin/bash
# GGUF KLD vs the BF16 GGUF base on data/eval_text.txt with llama-perplexity (ctx 2048, whole file; --kl-divergence ignores --chunks).
#   tools/gguf_kld.sh base                         -> writes the BF16 logits base ($KLD_BASE, about 21 GB)
#   tools/gguf_kld.sh NAME MODEL.gguf [auto|q4|q8] [BIN]
# auto = llama.cpp's default activation precision for the file (an untagged NVFP4 file -> W4A4 MMQ on Blackwell for prompt
# batches; prec_a4=false tag or any other quant -> Q8_1); q8 = GGML_CUDA_MMQ_PREC=q8 (W4A8 everywhere).
# Writes logs/gguf_kld_<name>.log and appends one JSON line to results/gguf_kld.jsonl.
# Environment: LLAMA_BIN (llama.cpp bin dir), KLD_BASE (base .kld file), BF16_GGUF (official LFM3-1.7B-BF16.gguf), PY (python).
set -u
L=${LFM3_EVAL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
BIN_DEF=${LLAMA_BIN:-/d/AI/tools/llama.cpp-b12110}
BASE=${KLD_BASE:-'C:\AI\Models\lfm3\kld\bf16_eval.kld'}
BF16=${BF16_GGUF:-/c/AI/Models/lfm3/gguf/LFM3-1.7B-BF16.gguf}
PY=${PY:-python}
TXT=$L/data/eval_text.txt
COMMON="-c 2048 -b 2048 -ub 512 -ngl 99 -fa on -t 16"
mkdir -p "$L/logs" "$L/results"
if [ "$1" = base ]; then
  BIN=${2:-$BIN_DEF}
  "$BIN/llama-perplexity" -m "$BF16" -f "$TXT" $COMMON --kl-divergence-base "$BASE" > "$L/logs/gguf_kld_base_bf16.log" 2>&1
  echo "base rc=$?"; grep -E "Final estimate|tokenizing|chunks" "$L/logs/gguf_kld_base_bf16.log" | tail -3
  exit 0
fi
NAME=$1; MODEL=$2; PREC=${3:-auto}; BIN=${4:-$BIN_DEF}
LOG=$L/logs/gguf_kld_${NAME}.log
if [ "$PREC" != auto ]; then export GGML_CUDA_MMQ_PREC=$PREC; fi
"$BIN/llama-perplexity" -m "$MODEL" -f "$TXT" $COMMON --kl-divergence-base "$BASE" --kl-divergence > "$LOG" 2>&1 < /dev/null
"$PY" -I - "$LOG" "$NAME" "$MODEL" "$PREC" "$BIN" "$L" <<'EOF'
import json, os, re, sys, time
log, name, model, prec, binp, root = sys.argv[1:]
t = open(log, encoding="utf-8", errors="ignore").read()
def g(pat, cast=float):
    m = re.search(pat, t)
    return cast(m.group(1)) if m else None
row = {"name": name, "prec": prec, "bin": binp, "model": model, "bytes": os.path.getsize(model), "time": time.strftime("%Y-%m-%d %H:%M"),
       "ppl": g(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)"), "ppl_base": g(r"Mean PPL\(base\)\s*:\s*([0-9.]+)"),
       "kld_mean": g(r"Mean\s+KLD:\s*([0-9.]+)"), "kld_median": g(r"Median\s+KLD:\s*([0-9.]+)"),
       "kld_p99": g(r"99\.0%\s+KLD:\s*([0-9.]+)"), "kld_max": g(r"Maximum KLD:\s*([0-9.]+)"),
       "same_top_pct": g(r"Same top p:\s*([0-9.]+)"), "rms_dp_pct": g(r"RMS Δp\s*:\s*([0-9.]+)")}
open(os.path.join(root, "results", "gguf_kld.jsonl"), "a", encoding="utf-8").write(json.dumps(row) + "\n")
print(json.dumps({k: row[k] for k in ("name", "prec", "bytes", "kld_mean", "kld_median", "kld_p99", "same_top_pct", "ppl", "ppl_base")}))
EOF
