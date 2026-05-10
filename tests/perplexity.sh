#!/bin/bash
# Perplexity + KL Divergence test: gold standard for quantization quality
# Compares Q4_0 vs q8_0 KV cache using wikitext-2
set -euo pipefail

PROJECT_DIR="$HOME/Dev/2026/qwen-3.6-35b"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
MODEL="$PROJECT_DIR/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
RESULTS_DIR="$PROJECT_DIR/tests/results"
WIKI_FILE="$LLAMA_DIR/wikitext-2-raw/wiki.test.raw"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$WIKI_FILE" ]; then
    echo "Downloading wikitext-2..."
    cd "$LLAMA_DIR"
    if [ -f scripts/get-wikitext-2.sh ]; then
        bash scripts/get-wikitext-2.sh
    else
        mkdir -p wikitext-2-raw
        curl -L "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" \
            -o /tmp/wikitext-2.zip
        unzip -o /tmp/wikitext-2.zip -d wikitext-2-raw/
        rm /tmp/wikitext-2.zip
    fi
    cd "$PROJECT_DIR"
fi

if [ ! -f "$WIKI_FILE" ]; then
    echo "ERROR: wikitext-2 not found at $WIKI_FILE"
    echo "Try manually: cd $LLAMA_DIR && bash scripts/get-wikitext-2.sh"
    exit 1
fi

NCMOE=${NCMOE:-25}
THREADS=${THREADS:-16}
PERPLEXITY="$LLAMA_DIR/build/bin/llama-perplexity"

echo "=== Perplexity + KL Divergence Test ==="
echo "Model: $(basename $MODEL)"
echo "Dataset: wikitext-2"
echo ""

# Step 1: q8_0 baseline (save logits for KL divergence)
echo "[1/2] Running q8_0 baseline (this takes ~30 min)..."
"$PERPLEXITY" \
    -m "$MODEL" \
    -f "$WIKI_FILE" \
    -ngl 99 \
    --n-cpu-moe "$NCMOE" \
    --flash-attn on \
    -ctk q8_0 -ctv q8_0 \
    -t "$THREADS" \
    --no-mmap \
    --save-all-logits "$RESULTS_DIR/baseline_q8.kld" \
    2>&1 | tee "$RESULTS_DIR/perplexity_q8_0.log"

echo ""

# Step 2: Q4_0 with KL divergence against baseline
echo "[2/2] Running Q4_0 with KL divergence comparison..."
"$PERPLEXITY" \
    -m "$MODEL" \
    -f "$WIKI_FILE" \
    -ngl 99 \
    --n-cpu-moe "$NCMOE" \
    --flash-attn on \
    -ctk q4_0 -ctv q4_0 \
    -t "$THREADS" \
    --no-mmap \
    --kl-divergence-base "$RESULTS_DIR/baseline_q8.kld" \
    --kl-divergence \
    2>&1 | tee "$RESULTS_DIR/perplexity_q4_0_kld.log"

echo ""
echo "=== Results saved to $RESULTS_DIR ==="
echo "Key metrics to check:"
echo "  - PPL delta (should be < 0.5)"
echo "  - KL divergence (should be < 0.05)"
echo "  - Same-top-p percentage (should be > 95%)"
