#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/Dev/2026/qwen-3.6-35b"
LLAMA_BENCH="$PROJECT_DIR/llama.cpp/build/bin/llama-bench"
MODEL="$PROJECT_DIR/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found at $MODEL"
    echo "Run setup.sh first or wait for download to complete."
    exit 1
fi

echo "=== Qwen3.6-35B-A3B Benchmark Suite ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo ""

# Test multiple ncmoe values to find sweet spot
for NCMOE in 27 25 23 21; do
    echo "--- Testing --n-cpu-moe $NCMOE ---"

    "$LLAMA_BENCH" \
        -m "$MODEL" \
        -ngl 99 \
        --n-cpu-moe "$NCMOE" \
        -fa 1 \
        -t 16 \
        -p 512 \
        -n 128 \
        -r 3 2>&1 || echo "  FAILED (likely OOM at ncmoe=$NCMOE)"

    echo ""
    # Show VRAM usage after test
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    echo ""
    sleep 2
done

echo "=== Benchmark Complete ==="
echo "Pick the ncmoe value with highest tok/s that doesn't OOM."
echo "Update NCMOE in run.sh accordingly."
