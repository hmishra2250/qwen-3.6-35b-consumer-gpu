#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/Dev/2026/qwen-3.6-35b"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"
MODEL_DIR="$PROJECT_DIR/models"

echo "=== Qwen3.6-35B-A3B Setup for RTX 4070 Max-Q (8GB) ==="
echo ""

# Step 1: Check prerequisites
echo "[1/4] Checking prerequisites..."
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake not found. Install with: sudo apt install cmake"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "ERROR: git not found."; exit 1; }
command -v nvcc >/dev/null 2>&1 || nvidia-smi >/dev/null 2>&1 || { echo "ERROR: CUDA not found."; exit 1; }
echo "  OK: cmake, git, CUDA detected"

# Step 2: Clone and build llama.cpp
echo ""
echo "[2/4] Building llama.cpp with CUDA..."
if [ ! -d "$LLAMA_DIR" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"
git pull --ff-only 2>/dev/null || true

rm -rf build
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_COMPRESSION_MODE=speed
cmake --build build --config Release -j$(nproc)

echo "  OK: llama.cpp built successfully"
echo "  Binary: $LLAMA_DIR/build/bin/llama-server"

# Step 3: Download model
echo ""
echo "[3/4] Downloading Qwen3.6-35B-A3B-UD-IQ3_XXS..."
mkdir -p "$MODEL_DIR"

MODEL_FILE="$MODEL_DIR/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"
if [ -f "$MODEL_FILE" ] && [ "$(stat --format=%s "$MODEL_FILE")" -gt 13000000000 ]; then
    echo "  Model already downloaded"
else
    # Use curl with HF token for reliable downloads (Xet protocol via hf CLI is slow)
    HF_TOKEN="${HF_TOKEN:-$(command -v hf >/dev/null 2>&1 && hf auth token 2>/dev/null || echo "")}"
    AUTH_HEADER=""
    if [ -n "$HF_TOKEN" ]; then
        AUTH_HEADER="-H \"Authorization: Bearer $HF_TOKEN\""
    fi
    echo "  Downloading 13.2 GB model file..."
    curl -L $AUTH_HEADER \
        "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" \
        -o "$MODEL_FILE" --progress-bar
fi

echo "  OK: Model downloaded"

# Step 4: Create launch script
echo ""
echo "[4/4] Creating launch script..."

cat > "$PROJECT_DIR/run.sh" << 'RUNEOF'
#!/bin/bash
PROJECT_DIR="$HOME/Dev/2026/qwen-3.6-35b"
LLAMA_SERVER="$PROJECT_DIR/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT_DIR/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf"

# Optimal settings for RTX 4070 Max-Q 8GB + 16GB RAM
# Q4_0 KV cache exploits Qwen3.6 hybrid attention (only 10/40 layers use KV)
NCMOE=${NCMOE:-25}
CTX=${CTX:-131072}
KV_TYPE=${KV_TYPE:-q4_0}
THREADS=${THREADS:-16}
PORT=${PORT:-8080}

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
RUNEOF

chmod +x "$PROJECT_DIR/run.sh"

echo "  OK: Launch script created at $PROJECT_DIR/run.sh"
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run:"
echo "  ./run.sh"
echo ""
echo "To tune ncmoe (higher = more on CPU = less VRAM but slower):"
echo "  NCMOE=25 ./run.sh    # Recommended (~43 tok/s, 128K ctx)"
echo "  NCMOE=23 ./run.sh    # Aggressive (~46 tok/s but tight VRAM)"
echo ""
echo "To use q8_0 KV cache instead (less context, slightly faster):"
echo "  KV_TYPE=q8_0 CTX=65536 ./run.sh"
echo ""
echo "To reduce context if RAM is tight:"
echo "  CTX=32768 ./run.sh"
