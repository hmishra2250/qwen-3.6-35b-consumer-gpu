#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Dev/2026/qwen-3.6-35b}"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
MODEL_DIR="$PROJECT_DIR/models"
MODEL_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
IQ4_MODEL="Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
IQ3_MODEL="Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
DOWNLOAD_IQ3="${DOWNLOAD_IQ3:-0}"

curl_hf() {
    local filename="$1"
    local min_bytes="$2"
    local target="$MODEL_DIR/$filename"

    if [ -f "$target" ] && [ "$(stat --format=%s "$target")" -gt "$min_bytes" ]; then
        echo "  OK: $filename already downloaded"
        return 0
    fi

    local curl_args=(-L)
    local hf_token="${HF_TOKEN:-}"
    if [ -z "$hf_token" ] && command -v hf >/dev/null 2>&1; then
        hf_token="$(hf auth token 2>/dev/null || true)"
    fi
    if [ -n "$hf_token" ]; then
        curl_args+=(-H "Authorization: Bearer $hf_token")
    fi

    echo "  Downloading $filename..."
    curl "${curl_args[@]}" \
        "https://huggingface.co/$MODEL_REPO/resolve/main/$filename" \
        -o "$target" --progress-bar
}

echo "=== Qwen3.6-35B-A3B Setup for RTX 4070 Max-Q (8GB) ==="
echo ""

echo "[1/4] Checking prerequisites..."
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake not found. Install with: sudo apt install cmake"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found."; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found."; exit 1; }
command -v nvcc >/dev/null 2>&1 || command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: CUDA/NVIDIA tooling not found."; exit 1; }
echo "  OK: cmake, git, curl, CUDA/NVIDIA tooling detected"

echo ""
echo "[2/4] Building llama.cpp with CUDA..."
if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
git pull --ff-only 2>/dev/null || true

cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_COMPRESSION_MODE=speed
cmake --build build --config Release -j"$(nproc)"

echo "  OK: llama.cpp built successfully"
echo "  Binary: $LLAMA_DIR/build/bin/llama-server"

echo ""
echo "[3/4] Downloading model weights..."
mkdir -p "$MODEL_DIR"

# IQ4_XS is the profile-ready recommended configuration from the final report.
curl_hf "$IQ4_MODEL" 17000000000

if [ "$DOWNLOAD_IQ3" = "1" ]; then
    curl_hf "$IQ3_MODEL" 13000000000
else
    echo "  Skipping IQ3_XXS fallback download. Set DOWNLOAD_IQ3=1 ./setup.sh to fetch it."
fi

echo ""
echo "[4/4] Verifying launch scripts..."
chmod +x "$PROJECT_DIR/run-iq4xs.sh" "$PROJECT_DIR/run.sh" "$PROJECT_DIR/test.sh" "$PROJECT_DIR/benchmark.sh" "$PROJECT_DIR"/tests/*.sh

echo "  OK: Launch scripts are executable"
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Recommended run:"
echo "  ./run-iq4xs.sh    # IQ4_XS, asymmetric KV, 128K context"
echo ""
echo "Fallback run if RAM is tight and you downloaded IQ3_XXS:"
echo "  ./run.sh"
echo ""
echo "Benchmark:"
echo "  API_TEMP=0.6 MAX_RETRIES=2 NO_THINK=1 python3 tests/swe_challenges.py local_iq4xs"
