# Checkpoint: Working State as of 2026-05-10

This document captures the exact known-good state so that if anything breaks
during future experimentation, you can return to this baseline and start over.

---

## 1. Hardware (Fixed)

```
GPU:    NVIDIA GeForce RTX 4070 Laptop GPU (Max-Q)
        8188 MiB VRAM, Ada Lovelace (sm_89), PCIe Gen4
CPU:    Intel Core i7-14700HX
        8 Performance cores (threads 0-15 with HT)
        6 Efficiency cores  (threads 16-27, no HT)
        28 threads total
RAM:    16 GB DDR5
OS:     Ubuntu, kernel 6.17.0-23-generic
Driver: NVIDIA 570.211.01 (CUDA 12.8 max capability)
CUDA:   Toolkit 12.6 (nvcc Build cuda_12.6.r12.6/compiler.35059454_0)
```

## 2. Software Versions (Pinned)

```
llama.cpp commit:  0b04728  ("sync : ggml")
Compiler:          g++ (Ubuntu 9.5.0-6ubuntu2.1) 9.5.0
cmake:             3.28.3
huggingface_hub:   1.14.0 (hf CLI)
```

### How llama.cpp was built

```bash
cd ~/Dev/2026/qwen-3.6-35b/llama.cpp

cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build --config Release -j$(nproc)
```

Flag explanations:
- `-DGGML_CUDA=ON` — enable NVIDIA CUDA backend
- `-DGGML_NATIVE=ON` — compile with -march=native for host CPU optimizations
- `-DCMAKE_BUILD_TYPE=Release` — optimized build, no debug symbols
- `-DCMAKE_CUDA_ARCHITECTURES=89` — target Ada Lovelace (RTX 4070) specifically;
  avoids compiling kernels for older GPUs, faster build and slightly faster binaries

Binaries produced:
- `build/bin/llama-server` — HTTP server with OpenAI-compatible API
- `build/bin/llama-bench` — benchmark tool
- `build/bin/llama-cli` — interactive CLI

## 3. Model File

```
File:     ~/Dev/2026/qwen-3.6-35b/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
Size:     13,211,155,424 bytes (13 GB / 12,597 MiB)
MD5:      18a639fe09a70c1d7ec99910d7a99c53
Source:   https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
Quant:    Unsloth Dynamic 2.0 IQ3_XXS (3-bit, intelligent layer upcasting)
```

### To re-download if deleted

```bash
# Option 1: curl with HF token (fast, reliable)
HF_TOKEN=$(hf auth token)
curl -L -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" \
    -o ~/Dev/2026/qwen-3.6-35b/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
    --progress-bar

# Option 2: hf CLI (uses Xet protocol — can be slow without auth)
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
    Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
    --local-dir ~/Dev/2026/qwen-3.6-35b/models

# Verify integrity after download
md5sum ~/Dev/2026/qwen-3.6-35b/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
# Expected: 18a639fe09a70c1d7ec99910d7a99c53
```

**Download lesson learned:** The `hf` CLI v1.14+ uses the Xet CDN protocol which
is extremely slow (~10 KB/s) without authentication. Always `hf auth login` first,
or use `curl -L` with `Authorization: Bearer` header for reliable downloads.

## 4. The Exact Server Command (Known Good)

```bash
~/Dev/2026/qwen-3.6-35b/llama.cpp/build/bin/llama-server \
    -m ~/Dev/2026/qwen-3.6-35b/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
    -ngl 99 \
    --n-cpu-moe 25 \
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

Or simply:

```bash
cd ~/Dev/2026/qwen-3.6-35b
./run.sh
```

## 5. Every Flag Explained

### Model loading

| Flag | Value | What it does | Why we chose it |
|------|-------|-------------|-----------------|
| `-m` | `...UD-IQ3_XXS.gguf` | Path to model weights | IQ3_XXS is 13.2 GB — fits in 16 GB RAM with room for KV cache and OS |
| `-ngl 99` | 99 | Number of GPU layers — 99 means "all of them" | Pushes all transformer layers to GPU initially; `--n-cpu-moe` then selectively pulls expert weights back to CPU |

### MoE offloading (THE critical flag)

| Flag | Value | What it does | Why we chose it |
|------|-------|-------------|-----------------|
| `--n-cpu-moe` | 25 | Keeps routed expert FFN weights from the first 25 layers on CPU RAM instead of VRAM | This is the key optimization. The model has 40 layers total. Layers 0-24's experts stay on CPU; layers 25-39's experts stay on GPU. Attention, dense FFN, and shared experts for ALL layers remain on GPU |

**How it works internally:**
1. `-ngl 99` loads everything to GPU
2. `--n-cpu-moe 25` then moves ONLY the routed expert weights (the 256 experts per layer)
   from layers 0-24 back to CPU RAM
3. During inference, when a token needs an expert from layers 0-24:
   - The token's small activation vector (~2 KB) is sent from GPU to CPU via PCIe
   - The CPU multiplies the activation against the expert weights in RAM
   - The result (~2 KB) is sent back to GPU
4. This is fast because only 8 of 256 experts activate per token, so the data
   crossing PCIe is tiny — the bottleneck is avoided

**Why 25 and not 23?**
- ncmoe=23 gave 46.1 tok/s but used 7635 MiB (only 180 MiB headroom)
- ncmoe=25 gives 44.3 tok/s but uses 7111 MiB (1075 MiB headroom)
- The 2 tok/s sacrifice buys stability under long-context fills and thermal throttling

### Attention optimization

| Flag | Value | What it does | Why we chose it |
|------|-------|-------------|-----------------|
| `--flash-attn` | `on` | Enables Flash Attention algorithm | ~30% VRAM savings on attention computation. Required when using quantized KV cache types. Must include explicit `on` — the bare flag alone does nothing |

### KV cache quantization

| Flag | Value | What it does | Why we chose it |
|------|-------|-------------|-----------------|
| `--cache-type-k` | `q8_0` | Quantize the Key cache to 8-bit integers | Halves KV cache memory vs FP16. At 35B+ model size, perplexity impact is negligible (~0.1%) |
| `--cache-type-v` | `q8_0` | Quantize the Value cache to 8-bit integers | Same benefit. Together with K quantization, this is what makes 64K context possible on 8 GB VRAM |

### Context and parallelism

| Flag | Value | What it does | Why we chose it |
|------|-------|-------------|-----------------|
| `--ctx-size` | 65536 | Maximum context window (tokens) | 64K tokens. Larger values (128K+) would overflow VRAM. KV cache at q8_0 for 64K ≈ 4 GB |
| `-np` | 1 | Number of parallel request slots | Single slot — each slot needs its own KV cache allocation. Multiple slots would multiply VRAM usage |

### CPU threading

| Flag | Value | What it does | Why we chose it |
|------|-------|-------------|-----------------|
| `-t` | 16 | Number of CPU threads for inference | 16 = the P-core threads (cores 0-7 with HyperThreading). The i7-14700HX's E-cores (threads 16-27) have lower IPC and add contention when mixed with P-cores for MoE expert computation |

### Memory management

| Flag | Value | What it does | Why we chose it |
|------|-------|-------------|-----------------|
| `--no-mmap` | (flag) | Disables memory-mapped file I/O — loads the full model into RAM via malloc instead | Avoids page fault overhead during generation. With mmap, the OS lazily loads pages on first access, causing stalls. --no-mmap pays the cost upfront (slower startup, ~30s) but generation is smoother. Reported 3+ tok/s improvement on constrained systems |

### Network

| Flag | Value | What it does | Why we chose it |
|------|-------|-------------|-----------------|
| `--host` | `127.0.0.1` | Listen address | Localhost only — not exposed to network. Use `0.0.0.0` to expose, or use Tailscale IP for remote access |
| `--port` | 8080 | Listen port | Default. Change if conflicting with other services |

## 6. Benchmark Results (Verified 2026-05-10)

### Throughput

| Test | Tokens | Wall time | Rate |
|------|--------|-----------|------|
| Short (128 tokens) | 128 | 3.10s | **41.3 tok/s** |
| Medium (1024 tokens) | 1024 | 22.96s | **44.6 tok/s** |
| Long (2048 tokens) | 2048 | 46.65s | **43.9 tok/s** |

Average sustained: **~43-45 tok/s**

### Resource usage during inference

| Resource | In Use | Total | Headroom |
|----------|--------|-------|----------|
| VRAM | 7111 MiB | 8188 MiB | 1077 MiB |
| RAM | ~13 GB | 16 GB | ~3 GB |
| GPU temp | 68-69°C | — | Safe |
| GPU power | 12W idle | 55W max | — |

### ncmoe comparison (this hardware)

| ncmoe | tok/s | VRAM (MiB) | Headroom | Stability |
|-------|-------|-----------|----------|-----------|
| 23 | 46.1 | 7635 | 180 MiB | Risky under load |
| **25** | **44.3** | **7111** | **1077 MiB** | **Stable (chosen)** |

### Comparison vs thread (RTX 3070 Ti 8GB)

| Config | 3070 Ti | 4070 Max-Q | Delta |
|--------|---------|------------|-------|
| ncmoe=25 | 40.9 | 44.3 | +8.3% |
| ncmoe=23 | 43.8 | 46.1 | +5.2% |

Improvement explained by RTX 4070's higher memory bandwidth (256 vs 192 GB/s).

## 7. Quality Verification

The model produces correct, coherent output:
- Python code with proper type hints, docstrings, edge case handling
- Multi-step reasoning chains (thinking mode active by default)
- No gibberish, hallucination artifacts, or quantization degradation observed

Thinking mode:
- Enabled by default (Qwen3.6 wraps reasoning in `<think>...</think>` tags)
- Visible output comes after the thinking block
- The API returns `reasoning_content` separately from `content`
- Thinking tokens count toward `completion_tokens` in usage stats

## 8. API Usage

The server exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1/`.

### Health check

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok"}
```

### Chat completion

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 512
    }'
```

### Streaming

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": true,
        "temperature": 0.6
    }'
```

### From Python (any OpenAI SDK client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen3.6",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.6,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

### Recommended sampling parameters

For coding tasks:
```json
{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
```

For general/creative tasks:
```json
{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
```

## 9. Project File Layout

```
~/Dev/2026/qwen-3.6-35b/
├── run.sh                  # Launch server (main entry point)
├── setup.sh                # Full setup from scratch (clone, build, download)
├── benchmark.sh            # Sweep ncmoe values to find sweet spot
├── test.sh                 # Smoke test: hit running server, measure speed
├── thread.md               # Original Twitter thread that inspired this setup
├── docs/
│   ├── GOAL.md             # System specs and performance targets
│   ├── RESEARCH.md         # All optimization research with sources
│   ├── SETUP.md            # Step-by-step manual setup instructions
│   ├── TUNING.md           # How to tune ncmoe, threads, context
│   ├── FINAL_CONFIG.md     # Flag-by-flag justification and memory budget
│   ├── RESULTS.md          # Benchmark numbers
│   └── CHECKPOINT_2026-05-10.md  # THIS FILE — full recovery snapshot
├── models/
│   └── Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf   # 13.2 GB model weights
└── llama.cpp/              # Built from source, commit 0b04728
    └── build/bin/
        ├── llama-server    # HTTP server binary
        ├── llama-bench     # Benchmark binary
        └── llama-cli       # Interactive CLI binary
```

## 10. Full Recovery Procedure

If everything breaks and you need to get back to this exact state:

### Step 1: Rebuild llama.cpp at this commit

```bash
cd ~/Dev/2026/qwen-3.6-35b/llama.cpp
git fetch origin
git checkout 0b04728
rm -rf build
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release -j$(nproc)
```

### Step 2: Verify model file

```bash
ls -l ~/Dev/2026/qwen-3.6-35b/models/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
# Must be 13,211,155,424 bytes (13 GB)
# MD5: 18a639fe09a70c1d7ec99910d7a99c53
```

If missing, re-download (see section 3 above).

### Step 3: Start server

```bash
cd ~/Dev/2026/qwen-3.6-35b
./run.sh
```

### Step 4: Verify

```bash
# Wait for "Loading model" to finish (check health endpoint)
curl http://127.0.0.1:8080/health
# Should return: {"status":"ok"}

# Quick speed check
time curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":64}'
# Should complete in ~2 seconds

# VRAM check
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Should show ~7100-7200 MiB
```

## 11. Known Issues and Gotchas

1. **CUDA 13.2 produces gibberish** with Qwen3.6 models — do not upgrade past 12.8
2. **First request is slow** (~5s) because the KV cache initializes; subsequent
   requests are normal speed
3. **--no-mmap makes startup slow** (~30s to preload 13 GB into RAM) — this is
   normal and the price of smoother inference
4. **Model uses 13 GB RAM** — with 16 GB total, only ~3 GB remains for the OS.
   Close heavy browser tabs before running. If system starts swapping, reduce
   context: `CTX=32768 ./run.sh`
5. **Thermal throttling on laptops** — sustained inference at 45 tok/s may heat
   the GPU to 75°C+. A cooling pad helps. If temps exceed 80°C, reduce threads:
   `THREADS=12 ./run.sh`
6. **Flash attention `on` is mandatory** — the bare `--flash-attn` flag without
   `on` does nothing. Always use `--flash-attn on`
7. **The `hf` CLI Xet downloads are unreliable** — always use `curl -L` with
   an HF token for downloads (see section 3)
8. **Don't use speculative decoding** — tested and shown to reduce throughput
   from 44 tok/s to ~11 tok/s on MoE models
9. **Don't use DFlash** — not stable for MoE architectures in mainline llama.cpp
10. **ncmoe too low (< 21) causes OOM cliff** — speed drops from 44 tok/s to
    <20 tok/s as the system starts swapping between VRAM and RAM

## 12. Future Optimization Paths

These were researched but not yet attempted:

### ik_llama.cpp fork
- ~1.5-2x faster MoE inference via fused operations
- Graph reuse (`-gr`) eliminates kernel launch overhead
- Build: same cmake flags, just clone from `ikawrakow/ik_llama.cpp`
- Risk: less tested, may break with model updates

### TurboQuant KV cache
- `--cache-type-k turbo3 --cache-type-v turbo3` → 4.9x cache compression
- Would enable 256K context instead of 64K
- Not yet in mainline llama.cpp — requires special build

### --mlock
- Pin model in RAM to prevent kernel paging
- Requires `sudo setcap cap_ipc_lock=+ep llama-server` or `ulimit -l unlimited`
- May improve stability under memory pressure

### Thread pinning
- `taskset -c 0-15 ./run.sh` to force P-cores only
- May help if OS scheduler routes work to E-cores

### Higher quality quantization
- UD-Q3_K_M (16.6 GB) would give better quality but might not fit in 16 GB RAM
  with 64K context. Would need `CTX=32768` as tradeoff.
