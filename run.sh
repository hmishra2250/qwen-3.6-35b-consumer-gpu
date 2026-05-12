#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Dev/2026/qwen-3.6-35b}"
LLAMA_SERVER="$PROJECT_DIR/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT_DIR/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

# IQ3_XXS fallback: smaller footprint, lower quality than run-iq4xs.sh.
# Q4_0 KV cache exploits Qwen3.6 hybrid attention (only 10/40 layers use KV).
# FA_ALL_QUANTS build flag enables efficient flash attention for Q4_0.
NCMOE=${NCMOE:-25}
CTX=${CTX:-131072}
KV_TYPE=${KV_TYPE:-q4_0}
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
    echo "Run: DOWNLOAD_IQ3=1 ./setup.sh"
    echo "Or: hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --local-dir $PROJECT_DIR/models"
    exit 1
fi

echo "=== Qwen3.6-35B-A3B Server (IQ3_XXS fallback) ==="
echo "  ncmoe: $NCMOE | ctx: $CTX | kv: $KV_TYPE/$KV_TYPE | threads: $THREADS | port: $PORT | reasoning_budget: $REASONING_BUDGET"
echo "  Model: $(basename "$MODEL")"
echo ""

exec "$LLAMA_SERVER" \
    -m "$MODEL" \
    -ngl 99 \
    --n-cpu-moe "$NCMOE" \
    --flash-attn on \
    --cache-type-k "$KV_TYPE" \
    --cache-type-v "$KV_TYPE" \
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
