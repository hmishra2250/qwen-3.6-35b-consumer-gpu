#!/bin/bash
PROJECT_DIR="$HOME/Dev/2026/qwen-3.6-35b"
LLAMA_SERVER="$PROJECT_DIR/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT_DIR/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

# Optimal settings for RTX 4070 Max-Q 8GB + 16GB RAM
# Q4_0 KV cache exploits Qwen3.6 hybrid attention (only 10/40 layers use KV)
# FA_ALL_QUANTS build flag enables efficient flash attention for Q4_0
NCMOE=${NCMOE:-25}
CTX=${CTX:-131072}
KV_TYPE=${KV_TYPE:-q4_0}
THREADS=${THREADS:-16}
PORT=${PORT:-8080}

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Run: hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --local-dir $PROJECT_DIR/models"
    exit 1
fi

echo "=== Qwen3.6-35B-A3B Server ==="
echo "  ncmoe: $NCMOE | ctx: $CTX | kv: $KV_TYPE | threads: $THREADS | port: $PORT"
echo "  Model: $(basename $MODEL)"
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
    --host 127.0.0.1 \
    --port "$PORT"
