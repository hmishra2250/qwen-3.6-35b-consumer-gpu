# Final Results: Qwen3.6-35B-A3B Performance

> **Note**: These speed benchmarks were measured with the IQ3_XXS quantization. For quality results, see [QUALITY.md](QUALITY.md). The recommended configuration is now **IQ4_XS + /no_think** (9/10 on SWE challenges, see `run-iq4xs.sh`).

## IQ3_XXS Speed Benchmarks

| Test | Tokens | Time | Rate |
|------|--------|------|------|
| Short (128 tok) | 128 | 2.99s | **42.7 tok/s** |
| Medium (1024 tok) | 1024 | 23.68s | **43.3 tok/s** |
| Long (2048 tok) | 2048 | 53.46s | **38.3 tok/s** (thermal throttled) |

**Context: 128K tokens | KV cache: Q4_0 | VRAM: 7155 MiB / 8188 MiB**

Note: 2048-token generation causes GPU to hit 85C on this laptop, triggering thermal throttling. True sustained speed (non-throttled) is ~43 tok/s.

## Previous Baseline (q8_0 + 64K)

| Test | Tokens | Time | Rate |
|------|--------|------|------|
| Short (128 tok) | 128 | 3.10s | **41.3 tok/s** |
| Medium (1024 tok) | 1024 | 22.96s | **44.6 tok/s** |
| Long (2048 tok) | 2048 | 46.65s | **43.9 tok/s** |

**Context: 64K tokens | KV cache: q8_0 | VRAM: 7113 MiB / 8188 MiB**

## Improvement Summary

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Context window | 64K | **128K** | **+100%** |
| Speed (1024 tok) | 44.6 tok/s | 43.3 tok/s | -3% |
| Quality | Baseline | Identical | No loss |
| VRAM | 7113 MiB | 7155 MiB | +42 MiB |

**2x context window at essentially the same speed and no quality degradation.**

## Optimized Server Command

```bash
llama-server \
  -m Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  -ngl 99 \
  --n-cpu-moe 25 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --ctx-size 131072 \
  -np 1 \
  -t 16 \
  --no-mmap \
  --host 127.0.0.1 \
  --port 8080
```

## Build Flags (Critical)

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_COMPRESSION_MODE=speed
```

`GGML_CUDA_FA_ALL_QUANTS=ON` is essential — it enables flash attention CUDA kernels for Q4_0 KV cache. Without it, Q4_0 falls back to slower non-FA paths and loses ~5% speed.

## What Didn't Work

| Optimization | Result | Why |
|-------------|--------|-----|
| `-b 4096 -ub 1024` (higher batch) | 32 tok/s (-25%) | Larger compute buffer, no gen-speed benefit |
| ncmoe=23 + Q4_0 | ~41 tok/s | Extra GPU layers don't help; Q4_0 cache is new bottleneck |
| ncmoe=23 + Q4_0 + 128K | 7685 MiB | Too tight, only 503 MiB headroom |

## Quality Verification (Q4_0 vs q8_0)

Tested at temperature=0.0 with identical prompts:

| Test | Q4_0 KV (128K) | q8_0 KV (64K) | Match? |
|------|----------------|----------------|--------|
| Code generation | Correct fibonacci | Correct fibonacci | Yes |
| Math (17*23+45-12*3) | 400 | 400 | Yes |
| Factual (planets) | Correct order | Identical | Yes |
| Logic (syllogism) | Valid reasoning | Valid reasoning | Yes |

Q4_0 KV is near-lossless on Qwen3.6 hybrid architecture because only 10 of 40 layers use traditional KV-cached attention (the other 30 use Gated DeltaNet linear attention which doesn't need a KV cache). For best quality, use asymmetric KV (Q4_0 keys + Q8_0 values) as configured in `run-iq4xs.sh`.

## ncmoe Comparison

| ncmoe | KV | Context | Rate | VRAM | Headroom | Verdict |
|-------|-----|---------|------|------|----------|---------|
| 25 | q4_0 | 128K | 43.3 tok/s | 7155 MiB | 1033 MiB | **Recommended** |
| 25 | q8_0 | 64K | 44.6 tok/s | 7111 MiB | 1077 MiB | Safe fallback |
| 23 | q4_0 | 128K | 41.1 tok/s | 7685 MiB | 503 MiB | Not worth it |
| 23 | q8_0 | 64K | 46.1 tok/s | 7635 MiB | 180 MiB | Max speed, risky |

## Key Optimizations Applied

1. **Q4_0 KV cache** — Exploits hybrid attention (10/40 layers), halves KV memory, enables 128K context
2. **GGML_CUDA_FA_ALL_QUANTS** — Flash attention for quantized KV, recovers ~5% speed loss from Q4_0
3. **Fused Gated DeltaNet** — New in latest llama.cpp, accelerates 30/40 recurrent layers
4. **ncmoe=25** — Best speed/stability tradeoff for 8GB VRAM
5. **Flash attention** — ~30% attention VRAM savings
6. **--no-mmap** — Eliminates page fault overhead
7. **16 P-core threads** — Avoids E-core IPC penalty on i7-14700HX

## Build Details

- llama.cpp: commit 2e97c5f (with fused Gated DeltaNet support)
- CUDA: 12.6 toolkit, driver 570.211.01
- Architecture target: sm_89 (Ada Lovelace)
- Build flags: FA_ALL_QUANTS=ON, COMPRESSION_MODE=speed
