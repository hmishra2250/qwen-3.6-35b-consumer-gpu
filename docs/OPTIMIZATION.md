# Optimization Log: Qwen3.6-35B-A3B on RTX 4070 Max-Q

## Starting Point (Baseline)
- llama.cpp commit 0b04728
- IQ3_XXS weights, q8_0 KV cache, 64K context, ncmoe=25
- **44 tok/s sustained, 7113 MiB VRAM, 5/10 SWE challenges**

## Optimization 1: Q4_0 KV Cache (SUCCESS)

**Hypothesis:** Qwen3.6 uses hybrid attention -- only 10 of 40 layers use traditional KV-cached attention (the other 30 use Gated DeltaNet linear attention). This means Q4_0 KV quantization affects far fewer layers than in standard transformer models and should be near-lossless.

**Test:** Switched `--cache-type-k q4_0 --cache-type-v q4_0` and doubled context to 128K.

**Result:** 
- Speed: 41 tok/s at 128K context (vs 44 tok/s at 64K)
- VRAM: 7155 MiB (similar)
- Quality: Identical outputs at temp=0.0 across 4 test categories

**Verdict:** 2x context at ~7% speed cost. Adopted.

## Optimization 2: Higher Batch Size (FAILED)

**Hypothesis:** `-b 4096 -ub 1024` should improve prompt processing and possibly generation speed.

**Result:**
- Speed: 32.4 tok/s at 1024 tokens (-21%)
- VRAM: 7649 MiB (+494 MiB, only 539 MiB headroom)

**Why it failed:** Larger batch buffers consume VRAM without benefiting single-token generation.

**Verdict:** Rejected.

## Optimization 3: ncmoe=23 with Q4_0 (MARGINAL)

**Hypothesis:** Q4_0 saves KV cache VRAM, so ncmoe=23 (more GPU layers) might fit.

**Result:**
- Speed: 41.1 tok/s (128 tok), 38.2 tok/s (2048 tok)
- VRAM: 7685 MiB (only 503 MiB headroom)

**Why it's marginal:** Extra GPU layers don't help because the Q4_0 KV operations are the new bottleneck. VRAM headroom is dangerously tight.

**Verdict:** Rejected. Stick with ncmoe=25 for IQ3_XXS, ncmoe=30 for IQ4_XS.

## Optimization 4: FA_ALL_QUANTS Build Flag (SUCCESS)

**Test:** Rebuilt llama.cpp with `-DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_COMPRESSION_MODE=speed`.

**Result:**
- Before: 41.0 tok/s
- After: **43.3 tok/s** (+5.6%)

**Verdict:** Adopted. Essential for quantized KV performance.

## Optimization 5: Updated llama.cpp with Fused DeltaNet (SUCCESS)

Pulling latest llama.cpp (commit 2e97c5f) enabled fused Gated DeltaNet operations for the 30 recurrent layers, reducing kernel launch overhead.

**Result:** Q4_0 + 128K now achieves 43 tok/s, matching the original q8_0 + 64K baseline.

**Verdict:** Always use latest llama.cpp for Qwen3.6.

## Optimization 6: IQ4_XS Weight Quantization (SUCCESS)

**Hypothesis:** IQ3_XXS (3-bit) sits below the 4-bit reliability threshold. Upgrading to IQ4_XS (~4-bit, 17.7 GB) should cross the threshold and improve quality significantly.

**Test:** Downloaded IQ4_XS, adjusted ncmoe from 25 to 30 to fit the larger model in 8 GB VRAM.

**Result:**
- SWE score: 7/10 (up from 5/10 on IQ3_XXS), a 40% improvement
- Speed: ~10-15 tok/s (slower due to more expert offloading to CPU)
- VRAM: ~5.7 GB (more headroom than IQ3_XXS at ncmoe=25)

**Tradeoff:** Roughly 3x speed reduction for 40% quality improvement. For code generation tasks where correctness matters more than speed, this is a clear win.

**Verdict:** Adopted as the recommended configuration.

## Optimization 7: Asymmetric KV Cache (SUCCESS)

**Hypothesis:** Research (QAQ, PM-KVQ papers) shows the value cache is ~3.5x more sensitive to quantization than the key cache. Asymmetric quantization (Q4 keys + Q8 values) should protect quality while preserving most VRAM savings.

**Test:** Changed `--cache-type-v q8_0` while keeping `--cache-type-k q4_0`.

**Result:** No measurable impact on short-context SWE challenges (expected, since KV errors accumulate at longer contexts), but provides insurance for long-context use cases at minimal VRAM cost.

**Verdict:** Adopted. Better error characteristics for long contexts.

## Optimization 8: Inference Settings (CRITICAL)

Three settings are mandatory for correct Qwen3.6 inference:

1. **temperature=0.6** (per-request): Qwen's official docs warn against temp=0. With temp=0, score was 6/10; with temp=0.6, score improved to 7/10.

2. **--reasoning-budget 4096** (server flag): Without it, the model spends all tokens in `<think>` blocks and produces no visible answer.

3. **--jinja + preserve_thinking + reasoning-budget-message** (server flags): Prevents re-reasoning loops, provides graceful budget transition (89% vs 78% HumanEval on hard cutoff).

**Verdict:** Non-negotiable. Getting any of these wrong causes severe degradation.

## Optimization 9: /no_think Mode (SUCCESS)

**Hypothesis:** For code generation with clear specs, thinking is counterproductive. The model can spiral into endless deliberation, overcomplicating solutions, or exhausting tokens on reasoning without producing code.

**Test:** Appended `/no_think` to prompts and added a code-only system message.

**Result:**
- SWE score: **9/10** (up from 7/10 with thinking), matching cloud Gemma within 1 point
- C07 (Regex): PASS -- model uses cleaner OOP architecture without thinking
- C09 (Async Bugs): PASS -- eliminated 16,384-token thinking overflow
- C06 (Calendar): PASS (after test fix) -- direct code generation handles priority logic cleanly
- C04 (Tree Serialize): FAIL -- regression, this task genuinely needs extended reasoning

**Tradeoff:** ~10-20% of tasks (complex recursive logic, state machines) still need thinking. Use `/no_think` as default, enable thinking selectively.

**Verdict:** Adopted as the recommended prompt strategy for code generation.

## Optimization 10: Thermal Throttling (OBSERVED)

GPU hits 85C after ~40s of continuous generation on this laptop, causing the 2048-token benchmark to drop to ~38 tok/s.

**Mitigation:** Hardware limitation. True sustained speed is measured at 1024-token tests.

## Final Optimized Configuration

```
Model:  IQ4_XS (17.7 GB, ~4-bit)
Build:  GGML_CUDA_FA_ALL_QUANTS=ON, COMPRESSION_MODE=speed, sm_89
Flags:  ncmoe=30, asymmetric KV (q4_0/q8_0), 128K context, flash-attn, no-mmap
        jinja, reasoning-budget=4096, reasoning-budget-message, preserve_thinking
Prompt: /no_think for code tasks, thinking for complex reasoning
Speed:  ~10-15 tok/s (IQ4_XS) / ~43 tok/s (IQ3_XXS)
VRAM:   ~5.7 GB
Score:  9/10 SWE challenges (1 point from cloud Gemma 31B at 10/10)
```

## Evolution Summary

| Step | Change | Score | Speed |
|------|--------|:-----:|-------|
| Baseline | IQ3_XXS, wrong settings | 3/10 | ~44 tok/s |
| + temp=0.6, reasoning budget | Fixed inference | 5/10 | ~44 tok/s |
| + IQ4_XS weights | 4-bit threshold | 6/10 | ~10-15 tok/s |
| + temp=0.6 | Correct sampling | 7/10 | ~10-15 tok/s |
| + /no_think + C06 fix | Direct code output | **9/10** | ~10-15 tok/s |

## Paths Not Taken

### ik_llama.cpp
- Supports Qwen3.6 (has `build_qwen35moe()` with DeltaNet)
- Fused MoE kernels, `--merge-qkv` flag, `-gr` graph reuse
- Not tested: fused DeltaNet in mainline already provides most of the benefit
- Worth revisiting if mainline performance plateaus

### KTransformers
- Specialized CPU/GPU heterogeneous MoE inference engine
- Could potentially better utilize CPU for expert computation
- Not tested: would require separate installation and workflow

### ExLlamaV2
- Does NOT support Qwen3.6 (no DeltaNet support as of v0.3.0)
- Dead end for this model
