# Optimization Log: Qwen3.6-35B-A3B on RTX 4070 Max-Q

## Starting Point (Baseline)
- llama.cpp commit 0b04728
- q8_0 KV cache, 64K context, ncmoe=25
- **44 tok/s sustained, 7113 MiB VRAM**

## Optimization 1: Q4_0 KV Cache (SUCCESS)

**Hypothesis:** Qwen3.6 uses hybrid attention -- only 10 of 40 layers use traditional KV-cached attention (the other 30 use Gated DeltaNet linear attention). This means Q4_0 KV quantization affects far fewer layers than in standard transformer models and should be lossless.

**Test:** Switched `--cache-type-k q4_0 --cache-type-v q4_0` and doubled context to 128K.

**Result:** 
- Speed: 41 tok/s at 128K context (vs 44 tok/s at 64K)
- VRAM: 7155 MiB (similar)
- Quality: Identical outputs at temp=0.0 across 4 test categories

**Verdict:** 2x context at ~7% speed cost. Adopted.

## Optimization 2: Higher Batch Size (FAILED)

**Hypothesis:** `-b 4096 -ub 1024` should improve prompt processing and possibly generation speed.

**Test:** Added batch size flags.

**Result:**
- Speed: 32.4 tok/s at 1024 tokens (-21%)
- VRAM: 7649 MiB (+494 MiB, only 539 MiB headroom)

**Why it failed:** Larger batch buffers consume VRAM without benefiting single-token generation (batch size only helps prefill). The increased compute buffer also pushed VRAM close to limits.

**Verdict:** Rejected. Default batch size is optimal for single-user inference.

## Optimization 3: ncmoe=23 with Q4_0 (MARGINAL)

**Hypothesis:** Q4_0 saves KV cache VRAM, so ncmoe=23 (more GPU layers) might fit with enough headroom.

**Test:** Combined ncmoe=23 with Q4_0 + 128K.

**Result:**
- Speed: 41.1 tok/s (128 tok), 38.2 tok/s (2048 tok)
- VRAM: 7685 MiB (only 503 MiB headroom)

**Why it's marginal:** Extra GPU layers don't help because the Q4_0 KV cache operations are the new bottleneck, not expert offloading. And the VRAM headroom is dangerously tight.

**Verdict:** Rejected. Stick with ncmoe=25.

## Optimization 4: FA_ALL_QUANTS Build Flag (SUCCESS)

**Hypothesis:** `GGML_CUDA_FA_ALL_QUANTS=ON` enables flash attention CUDA kernels for all KV quantization types including Q4_0. Without it, Q4_0 KV falls back to slower non-flash-attention paths.

**Test:** Rebuilt llama.cpp with `-DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_COMPRESSION_MODE=speed`.

**Result:**
- Before: 41.0 tok/s at 1024 tokens
- After: **43.3 tok/s** at 1024 tokens (+5.6%)
- No VRAM difference

**Why it works:** The flash attention CUDA kernel for Q4_0 performs the KV dequantization inline during attention computation, avoiding a separate dequantization pass.

**Verdict:** Adopted. Essential for Q4_0 KV performance.

## Optimization 5: Updated llama.cpp with Fused DeltaNet (SUCCESS)

**Observation:** Pulling latest llama.cpp (commit 2e97c5f) gave us:
- `fused Gated Delta Net (autoregressive) enabled`
- `fused Gated Delta Net (chunked) enabled`

This fuses the DeltaNet operations for the 30 recurrent layers, reducing kernel launch overhead.

**Result:** Combined with FA_ALL_QUANTS, Q4_0 + 128K now achieves 43 tok/s -- essentially matching the original q8_0 + 64K baseline.

**Verdict:** Always use latest llama.cpp for Qwen3.6.

## Optimization 6: Thermal Throttling (OBSERVED)

**Finding:** GPU hits 85C after ~40s of continuous generation on this laptop, causing the 2048-token benchmark to drop to ~38 tok/s. The 1024-token benchmark (23s) stays under 82C and achieves full speed.

**Mitigation:** This is a hardware limitation. For sustained long-generation workloads, consider:
- Laptop cooling pad
- Reducing ambient temperature
- The true sustained speed is ~43 tok/s (1024-token test)

## Final Optimized Configuration

```
Build: GGML_CUDA_FA_ALL_QUANTS=ON, COMPRESSION_MODE=speed, sm_89
Flags: ncmoe=25, q4_0 KV, 128K context, flash-attn, no-mmap, 16 threads
Speed: ~43 tok/s sustained (128K context)
VRAM:  7155 MiB (1033 MiB headroom)
```

## Paths Not Taken

### ik_llama.cpp
- Supports Qwen3.6 (has `build_qwen35moe()` with DeltaNet)
- Fused MoE kernels, `--merge-qkv` flag, `-gr` graph reuse
- Not tested due to: fused DeltaNet in mainline already provides most of the benefit
- Worth revisiting if mainline performance plateaus

### KTransformers
- Specialized CPU/GPU heterogeneous MoE inference engine
- Could potentially better utilize CPU for expert computation
- Not tested: would require separate installation and workflow

### ExLlamaV2
- Does NOT support Qwen3.6 (no DeltaNet support as of v0.3.0)
- Dead end for this model
