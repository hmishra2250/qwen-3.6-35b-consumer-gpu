# Qwen3.6-35B-A3B on Consumer Hardware

Running a 35B-parameter MoE model on a laptop GPU (RTX 4070 Max-Q, 8GB VRAM) with **128K context**. A practical guide to what works, what breaks, and how to get the best quality from aggressive quantization.

## Key Results

| Config | Score | Context | Speed | VRAM | Notes |
|--------|:-----:|---------|-------|------|-------|
| **IQ4_XS + /no_think** (best) | **9/10** | 128K | ~10-15 t/s | 5.7 GB + RAM | 1 point from cloud Gemma |
| IQ4_XS + thinking | 7/10 | 128K | ~10-15 t/s | 5.7 GB + RAM | Better on complex reasoning |
| IQ3_XXS (original) | 5/10 | 128K | ~10-17 t/s | 3.8 GB + RAM | Below 4-bit threshold |
| gemma-4-31b-it (cloud API) | 10/10 | -- | -- | -- | Cloud baseline for comparison |

Tested on 10 hard SWE coding challenges (algorithms, data structures, concurrency, system design) with temperature=0.6 and up to 2 retries per challenge.

## Hardware

- GPU: NVIDIA RTX 4070 Max-Q (8 GB VRAM)
- CPU: Intel i7-14700HX (8P + 6E cores)
- RAM: 16 GB DDR5
- CUDA: 12.6

## Quick Start

```bash
# 1. Setup (clone llama.cpp, build with CUDA, download model)
./setup.sh

# 2. Run the server (IQ3_XXS, original config)
./run.sh

# 3. Run the server (IQ4_XS, recommended for quality)
./run-iq4xs.sh

# 4. Test it
./test.sh

# 5. Run SWE benchmark suite (with /no_think for best results)
API_TEMP=0.6 MAX_RETRIES=2 NO_THINK=1 python3 tests/swe_challenges.py my_label

# 6. Use it (OpenAI-compatible API)
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "temperature": 0.6}'
```

## Two Quantization Options

### IQ4_XS (Recommended)

```bash
# Download (~17 GB)
hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --local-dir models

# Run (asymmetric KV: Q4 keys + Q8 values)
./run-iq4xs.sh
```

- 9/10 on SWE challenges with /no_think (7/10 with thinking), up from 5/10 on IQ3_XXS
- Above the 4-bit reliability threshold for reasoning tasks
- Fits 8 GB VRAM + 16 GB RAM with `--n-cpu-moe 30`
- Asymmetric KV cache protects the more sensitive value cache

### IQ3_XXS (Maximum context, lower quality)

```bash
# Download (~13 GB)
hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --local-dir models

# Run
./run.sh
```

- 5/10 on SWE challenges
- Smaller footprint, more RAM headroom
- Below the 4-bit threshold: tool calling and complex reasoning degrade

## Critical Configuration

Three settings are mandatory for correct Qwen3.6 inference. Getting any of them wrong causes severe degradation:

| Setting | Value | What Happens Without It |
|---------|-------|------------------------|
| `--reasoning-budget 4096` | Server flag | Model spends all tokens in `<think>` blocks, produces no visible answer |
| `--reasoning-budget-message` | Server flag | Hard cutoff instead of graceful transition when budget is hit |
| `--jinja` + `preserve_thinking` | Server flag | Model re-reasons from scratch each turn, wasting tokens |
| `temperature=0.6` | Per-request | Repetition loops, stuck generation (official docs warn against temp=0) |
| `/no_think` in prompt | For code tasks | Model reasons excessively in visible output instead of producing code |

## Optimizations Applied

### 1. MoE Expert Offloading (`--n-cpu-moe`)
Keeps attention layers on GPU while offloading expert FFN weights to CPU RAM. The PCIe transfer of small activation vectors is negligible compared to the VRAM savings.

### 2. Asymmetric KV Cache (`--cache-type-k q4_0 --cache-type-v q8_0`)
Qwen3.6 uses **hybrid attention**: only 10/40 layers use traditional KV-cached attention. The other 30 use Gated DeltaNet (linear attention) which needs no KV cache. This makes KV quantization far less impactful than on standard transformers. The value cache is ~3.5x more sensitive to quantization than the key cache, so asymmetric quantization (Q4 keys, Q8 values) is the optimal VRAM-quality tradeoff.

### 3. Flash Attention for All Quants
Built with `GGML_CUDA_FA_ALL_QUANTS=ON` to enable native CUDA flash attention kernels for quantized KV cache.

### 4. Fused Gated DeltaNet
Latest llama.cpp fuses the DeltaNet operations for the 30 recurrent layers, reducing kernel launch overhead.

### 5. No Memory Mapping
`--no-mmap` preloads the entire model into RAM, eliminating page fault overhead during generation.

## Build Flags

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_COMPRESSION_MODE=speed
```

## SWE Benchmark Results

10 hard coding challenges. Best results per config with correct settings (temp=0.6, reasoning_budget=4096, up to 2 retries).

| # | Challenge | IQ4_XS /no_think | IQ4_XS think | IQ3_XXS | gemma-31b |
|---|-----------|:------:|:------:|:------:|:---------:|
| C01 | Count of Range Sum | PASS | PASS | PASS | PASS |
| C02 | Burst Balloons | PASS | PASS | PASS | PASS |
| C03 | Matrix Fibonacci + Pisano | PASS | PASS | PASS | PASS |
| C04 | Tree Serialize/Deserialize | FAIL | PASS | FAIL | PASS |
| C05 | All Topological Sorts | PASS | PASS | PASS | PASS |
| C06 | Calendar Interval Merging | PASS | FAIL | FAIL | PASS |
| C07 | Mini Regex Engine | PASS | FAIL | FAIL | PASS |
| C08 | Consistent Hash Ring | PASS | PASS | FAIL | PASS |
| C09 | Async Queue Bugs | PASS | FAIL | FAIL | PASS |
| C10 | LRU Cache with TTL | PASS | PASS | PASS | PASS |
| | **Total** | **9/10** | 7/10 | 5/10 | **10/10** |

## Configuration Profiles

```bash
# IQ4_XS, 128K context, asymmetric KV (recommended)
./run-iq4xs.sh

# IQ3_XXS, 128K context, symmetric Q4 KV
./run.sh

# IQ3_XXS, 64K context, q8_0 KV
KV_TYPE=q8_0 CTX=65536 ./run.sh

# IQ3_XXS, 32K context, safe low-VRAM
CTX=32768 ./run.sh
```

## Documentation

| Doc | Contents |
|-----|----------|
| [findings.html](docs/findings.html) | Comprehensive findings report with community evidence |
| [QUALITY.md](docs/QUALITY.md) | Quality verification methodology and results |
| [RESULTS.md](docs/RESULTS.md) | Benchmark numbers, speed comparisons |
| [FINAL_CONFIG.md](docs/FINAL_CONFIG.md) | Complete flag justification, memory budget |
| [OPTIMIZATION.md](docs/OPTIMIZATION.md) | What worked, what didn't, and why |
| [RESEARCH.md](docs/RESEARCH.md) | All research findings with sources |
| [TUNING.md](docs/TUNING.md) | How to tune for your hardware |
| [SETUP.md](docs/SETUP.md) | Manual setup instructions |

## Why This Works

The key insight is that Qwen3.6-35B-A3B is not a standard transformer. Its hybrid architecture (Gated DeltaNet + sparse MoE) creates two orthogonal optimization opportunities:

1. **MoE sparsity** (only 3B of 35B params active per token) lets you offload most weights to CPU
2. **Hybrid attention** (only 10/40 layers use KV cache) makes aggressive KV quantization far less impactful

Combining these on consumer hardware hits a sweet spot where a $1,500 laptop runs a 35B model that would normally need 80GB+ of VRAM, at 128K context, scoring 9/10 on hard coding challenges -- one point from cloud-served Gemma 4 31B at full precision.

## License

Scripts and documentation: MIT. Model weights are subject to [Qwen's license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
