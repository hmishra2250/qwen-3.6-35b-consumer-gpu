#!/bin/bash
# Passkey retrieval test: stresses KV cache quality with long context
# The model must retrieve an exact number hidden in filler text
# This directly tests whether KV quantization loses information
set -euo pipefail

PROJECT_DIR="$HOME/Dev/2026/qwen-3.6-35b"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
MODEL="$PROJECT_DIR/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
RESULTS_DIR="$PROJECT_DIR/tests/results"
PASSKEY="$LLAMA_DIR/build/bin/llama-passkey"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$PASSKEY" ]; then
    echo "ERROR: llama-passkey not found. Rebuild llama.cpp."
    exit 1
fi

NCMOE=${NCMOE:-25}
THREADS=${THREADS:-16}

echo "=== Passkey Retrieval Test ==="
echo "Tests long-context KV cache integrity"
echo ""

for JUNK in 50 100 250 500; do
    echo "--- Junk level: $JUNK ---"

    echo "[q8_0] Running..."
    "$PASSKEY" \
        -m "$MODEL" \
        -ngl 99 \
        --n-cpu-moe "$NCMOE" \
        --flash-attn on \
        -ctk q8_0 -ctv q8_0 \
        -t "$THREADS" \
        --no-mmap \
        --junk "$JUNK" \
        2>&1 | tee "$RESULTS_DIR/passkey_q8_junk${JUNK}.log" | tail -5

    echo "[q4_0] Running..."
    "$PASSKEY" \
        -m "$MODEL" \
        -ngl 99 \
        --n-cpu-moe "$NCMOE" \
        --flash-attn on \
        -ctk q4_0 -ctv q4_0 \
        -t "$THREADS" \
        --no-mmap \
        --junk "$JUNK" \
        2>&1 | tee "$RESULTS_DIR/passkey_q4_junk${JUNK}.log" | tail -5

    echo ""
done

echo "=== Passkey Results ==="
echo "Compare pass/fail rates between q8_0 and q4_0 at each junk level."
echo "Degradation at higher junk levels indicates KV cache information loss."
echo "Results saved to $RESULTS_DIR/passkey_*.log"
