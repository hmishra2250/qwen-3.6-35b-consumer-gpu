#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Dev/2026/qwen-3.6-35b}"
LLAMA_SERVER="$PROJECT_DIR/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT_DIR/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"

# IQ4_XS: 17.7 GB on disk, above the 4-bit reliability threshold
# With --n-cpu-moe 30, ~11 GB experts on CPU, ~5.7 GB on GPU
# Asymmetric KV: Q4 keys (less sensitive) + Q8 values (more sensitive)
# Jinja templating enables preserve_thinking and enable_thinking controls
# Reasoning budget message provides graceful transition instead of hard cutoff
NCMOE=${NCMOE:-30}
CTX=${CTX:-131072}
THREADS=${THREADS:-16}
PORT=${PORT:-8080}
REASONING_BUDGET=${REASONING_BUDGET:-4096}
REASONING_MSG=${REASONING_MSG:-"I need to provide my answer now."}


if [ ! -x "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found or not executable at $LLAMA_SERVER"
    echo "Run: ./setup.sh"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Run: hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --local-dir $PROJECT_DIR/models"
    exit 1
fi

echo "=== Qwen3.6-35B-A3B Server (IQ4_XS) ==="
echo "  ncmoe: $NCMOE | ctx: $CTX | kv: q4_0/q8_0 | threads: $THREADS | port: $PORT | reasoning_budget: $REASONING_BUDGET"
echo "  Model: $(basename "$MODEL")"
echo ""

exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    -ngl 99 \
    --n-cpu-moe "$NCMOE" \
    --flash-attn on \
    --cache-type-k q4_0 \
    --cache-type-v q8_0 \
    --ctx-size "$CTX" \
    -np 1 \
    -t "$THREADS" \
    --no-mmap \
    --jinja \
    --reasoning-budget "$REASONING_BUDGET" \
    --reasoning-budget-message "$REASONING_MSG" \
    --reasoning-format deepseek \
    --chat-template-kwargs '{"preserve_thinking":true}' \
    --host 127.0.0.1 \
    --port "$PORT"
