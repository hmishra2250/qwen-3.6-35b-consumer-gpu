# Setup Guide: Qwen3.6-35B-A3B on RTX 4070 Max-Q (8GB)

## Prerequisites

- CUDA 12.8 (already installed)
- cmake, git, build-essential
- ~14GB free disk space for the model
- huggingface-cli (for downloading)

## Step 1: Build llama.cpp from Source

```bash
cd ~/Dev/2026/qwen-3.6-35b
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

Verify build:
```bash
./build/bin/llama-server --version
```

## Step 2: Download Model

```bash
# Install huggingface CLI if not present
pip install -U huggingface_hub[cli]

# Download UD-IQ3_XXS (13.2 GB, single file)
huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \
  --include "UD-IQ3_XXS/*" \
  --local-dir ~/Dev/2026/qwen-3.6-35b/models
```

## Step 3: System Preparation

```bash
# Allow mlock without root (prevents model from being swapped)
sudo setcap cap_ipc_lock=+ep ~/Dev/2026/qwen-3.6-35b/llama.cpp/build/bin/llama-server

# Or alternatively, set ulimit for current session
ulimit -l unlimited
```

## Step 4: Launch Server (Optimal Configuration)

```bash
~/Dev/2026/qwen-3.6-35b/llama.cpp/build/bin/llama-server \
  -m ~/Dev/2026/qwen-3.6-35b/models/UD-IQ3_XXS/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  -ngl 99 \
  --n-cpu-moe 23 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ctx-size 65536 \
  -np 1 \
  -t 16 \
  --no-mmap \
  --host 127.0.0.1 \
  --port 8080
```

### Flag Breakdown:
| Flag | Purpose |
|------|---------|
| `-ngl 99` | Push all layers to GPU |
| `--n-cpu-moe 23` | Keep experts from first 23 layers on CPU (sweet spot) |
| `--flash-attn on` | Enable flash attention (~30% VRAM savings) |
| `--cache-type-k q8_0` | Quantize K cache (halves KV memory) |
| `--cache-type-v q8_0` | Quantize V cache |
| `--ctx-size 65536` | 64K context window |
| `-np 1` | Single parallel slot (saves memory) |
| `-t 16` | 16 threads (P-cores only for best IPC) |
| `--no-mmap` | Preload to RAM (avoids page faults) |

## Step 5: Verify

```bash
# Check server is running
curl http://127.0.0.1:8080/health

# Check available models
curl http://127.0.0.1:8080/v1/models

# Test generation speed
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "messages": [{"role": "user", "content": "Write a quicksort in Python"}],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 512
  }'
```

## Step 6: Tuning

If VRAM is too tight (OOM or <30 tok/s):
- Increase ncmoe to 25: `--n-cpu-moe 25`
- Reduce context: `--ctx-size 32768`

If VRAM has headroom (check `nvidia-smi`):
- Decrease ncmoe to 21: `--n-cpu-moe 21` (more on GPU = faster)
- But watch for the cliff — below 21, performance collapses

## Benchmark Script

```bash
# Run the built-in benchmark
~/Dev/2026/qwen-3.6-35b/llama.cpp/build/bin/llama-bench \
  -m ~/Dev/2026/qwen-3.6-35b/models/UD-IQ3_XXS/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  -ngl 99 \
  --n-cpu-moe 23 \
  -fa 1 \
  -t 16
```
