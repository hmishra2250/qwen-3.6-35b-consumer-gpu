# Setup Guide: Qwen3.6-35B-A3B on RTX 4070 Max-Q (8GB)

## Prerequisites

- CUDA 12.6+ toolkit
- cmake, git, build-essential
- ~18 GB free disk space for the IQ4_XS model (or ~14 GB for IQ3_XXS)
- `hf` CLI (huggingface_hub)

## Step 1: Build llama.cpp from Source

```bash
cd ~/Dev/2026/qwen-3.6-35b
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_COMPRESSION_MODE=speed
cmake --build build --config Release -j$(nproc)
```

Verify build:
```bash
./build/bin/llama-server --version
```

The `GGML_CUDA_FA_ALL_QUANTS=ON` flag is critical: it enables flash attention CUDA kernels for quantized KV cache, recovering ~5% speed.

## Step 2: Download Model

Fast path: if you want the script to perform Step 1 and download IQ4_XS in one command, run:

```bash
./setup.sh
```

To also fetch the smaller IQ3_XXS fallback used by `run.sh`, run the setup script with:

```bash
DOWNLOAD_IQ3=1 ./setup.sh
```

Manual downloads, if you already built `llama.cpp` above and do not want to use `setup.sh`:

```bash
# Install hf CLI if not present
pip install -U huggingface_hub[cli]

# IQ4_XS (recommended, 17.7 GB, 9/10 SWE score)
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --local-dir models

# IQ3_XXS (alternative, 13.2 GB, 5/10 SWE score)
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  --local-dir models
```

## Step 3: Launch Server

```bash
# IQ4_XS with all optimizations (recommended)
./run-iq4xs.sh

# IQ3_XXS (smaller, lower quality)
./run.sh
```

The server will take a few minutes to load the model. Check readiness with:
```bash
curl http://127.0.0.1:8080/health
# Returns {"status":"ok"} when ready
```

## Step 4: Verify

```bash
# Test generation
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a quicksort in Python /no_think"}],
    "temperature": 0.6,
    "max_tokens": 512
  }'
```

## Step 5: Run Benchmarks

```bash
# Full SWE challenge suite (recommended: /no_think mode)
API_TEMP=0.6 MAX_RETRIES=2 NO_THINK=1 python3 tests/swe_challenges.py my_label

# Single challenge (use FILTER env var)
FILTER=C06 API_TEMP=0.6 NO_THINK=1 python3 tests/swe_challenges.py test

# Quick smoke test
./test.sh
```

## Tuning

If VRAM is too tight (OOM):
- For IQ4_XS: increase ncmoe (`NCMOE=32 ./run-iq4xs.sh`) to move more experts to CPU
- Reduce context: `CTX=65536 ./run-iq4xs.sh`

If RAM is pressured (swap thrashing):
- Switch to IQ3_XXS after downloading it: `DOWNLOAD_IQ3=1 ./setup.sh && ./run.sh` (13.2 GB vs 17.7 GB)
- Reduce context: `CTX=65536 ./run.sh`

See [TUNING.md](TUNING.md) for detailed tuning guidance.
