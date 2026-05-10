# Qwen3.6-35B-A3B on Consumer Hardware

Running a 35B-parameter MoE model at **43+ tok/s** on a laptop GPU (RTX 4070 Max-Q, 8GB VRAM) with **128K context** and zero quality loss.

## What This Is

A complete, reproducible setup for running [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B total / 3B active parameters, 256 experts) on consumer hardware using [llama.cpp](https://github.com/ggml-org/llama.cpp) with aggressive but lossless optimizations.

## Key Results

| Metric | Value |
|--------|-------|
| Generation speed | **43 tok/s** sustained |
| Context window | **128K tokens** |
| VRAM usage | 7.2 GB / 8.2 GB |
| RAM usage | ~13 GB / 16 GB |
| Quality loss | **None** (verified) |

## Hardware

- GPU: NVIDIA RTX 4070 Max-Q (8 GB VRAM)
- CPU: Intel i7-14700HX (8P + 6E cores)
- RAM: 16 GB DDR5
- CUDA: 12.6

## Quick Start

```bash
# 1. Setup (clone llama.cpp, build with CUDA, download model)
./setup.sh

# 2. Run the server
./run.sh

# 3. Test it
./test.sh

# 4. Use it (OpenAI-compatible API)
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "temperature": 0.6}'
```

## Optimizations Applied

### 1. MoE Expert Offloading (`--n-cpu-moe 25`)
Keeps attention layers on GPU while offloading expert FFN weights from 25/40 layers to CPU RAM. The PCIe transfer of small activation vectors is negligible compared to the VRAM savings.

### 2. Q4_0 KV Cache (lossless on this model)
Qwen3.6 uses **hybrid attention**: only 10/40 layers use traditional KV-cached attention. The other 30 use Gated DeltaNet (linear attention) which doesn't need a KV cache. This means Q4_0 KV quantization affects far fewer layers than in standard transformers, making it effectively lossless while halving KV cache memory and enabling 128K context.

### 3. Flash Attention for All Quants
Built with `GGML_CUDA_FA_ALL_QUANTS=ON` to enable native CUDA flash attention kernels for Q4_0 KV cache, recovering the ~5% speed penalty from Q4_0 dequantization.

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

`GGML_CUDA_FA_ALL_QUANTS=ON` is the single most impactful build flag for Q4_0 KV performance.

## Configuration Profiles

```bash
# Default: 128K context, 43 tok/s (recommended)
./run.sh

# High-speed: 64K context, 46 tok/s
KV_TYPE=q8_0 CTX=65536 ./run.sh

# Max speed, tight VRAM: 64K context, 46 tok/s, risky
KV_TYPE=q8_0 CTX=65536 NCMOE=23 ./run.sh

# Low VRAM: 32K context, safe
CTX=32768 ./run.sh
```

## Quality Verification

Q4_0 vs q8_0 KV cache tested with:
- Perplexity (wikitext-2)
- Passkey retrieval (long-context stress)
- Multi-step calculus, Einstein's riddle, code generation, physics derivations
- All at temperature=0.0 for deterministic comparison

See [docs/QUALITY.md](docs/QUALITY.md) for full results.

## Documentation

| Doc | Contents |
|-----|----------|
| [RESULTS.md](docs/RESULTS.md) | Benchmark numbers, speed comparisons |
| [FINAL_CONFIG.md](docs/FINAL_CONFIG.md) | Complete flag justification, memory budget |
| [OPTIMIZATION.md](docs/OPTIMIZATION.md) | What worked, what didn't, and why |
| [RESEARCH.md](docs/RESEARCH.md) | All research findings with sources |
| [TUNING.md](docs/TUNING.md) | How to tune for your hardware |
| [SETUP.md](docs/SETUP.md) | Manual setup instructions |
| [QUALITY.md](docs/QUALITY.md) | Q4_0 vs q8_0 quality test results |
| [CHECKPOINT_2026-05-10.md](docs/CHECKPOINT_2026-05-10.md) | Recovery guide with exact system state |

## Why This Works

The key insight is that Qwen3.6-35B-A3B is not a standard transformer. Its hybrid architecture (Gated DeltaNet + sparse MoE) creates two orthogonal optimization opportunities:

1. **MoE sparsity** (only 3B of 35B params active per token) lets you offload most weights to CPU
2. **Hybrid attention** (only 10/40 layers use KV cache) makes aggressive KV quantization lossless

Most optimization guides treat these independently. Combining them on the right hardware hits a sweet spot where a $1,500 laptop runs a model that would normally need 80GB+ of VRAM.

## License

Scripts and documentation: MIT. Model weights are subject to [Qwen's license](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
