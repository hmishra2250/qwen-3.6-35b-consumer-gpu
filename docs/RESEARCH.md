# Research Findings: Optimizing Qwen3.6-35B-A3B for 8GB VRAM

## 1. The -ncmoe Flag (Critical Optimization)

The `--n-cpu-moe N` flag keeps MoE expert weights from the first N layers in CPU RAM
instead of VRAM. This is THE key optimization for 8GB cards.

### How it works:
- With `-ngl 99`, all layers are initially assigned to GPU
- `--n-cpu-moe N` then moves expert FFN weights from layers 0..N-1 back to CPU
- Attention, dense FFN, and shared experts STAY on GPU (used every token)
- Only the routed expert activations cross PCIe (tiny vectors, not full weights)

### Thread benchmarks (RTX 3070 Ti, 8GB, IQ3_XXS):
| -ncmoe value | Speed    | VRAM Used | RAM Used |
|-------------|----------|-----------|----------|
| none        | 8.7 t/s  | 7.8 GB    | 13.6 GB  |
| 35          | 27.5 t/s | 4.3 GB    | 12.1 GB  |
| 30          | 32.5 t/s | 5.6 GB    | 12 GB    |
| 25          | 40.9 t/s | 6.9 GB    | 12 GB    |
| 23          | 43.8 t/s | 7.4 GB    | 12.2 GB  |
| 21          | 38.6 t/s | 7.8 GB    | 12.4 GB  |
| 19          | 19.8 t/s | -         | -        |

**Sweet spot: ncmoe 23-25** — maximizes GPU utilization without OOM.
At ncmoe 21+, VRAM is overfull and starts swapping, killing performance.

### For our RTX 4070 Max-Q (8GB):
- Start testing at ncmoe 25 (safest)
- Try ncmoe 23 (potentially fastest)
- The 4070 has higher bandwidth than 3070 Ti, expect slightly better numbers
- Target: leave ~600-800MB VRAM headroom for stability

## 2. Quantization Choice

### Best options for 8GB VRAM + 16GB RAM:

| Quant         | File Size | Quality   | Fits? |
|--------------|-----------|-----------|-------|
| UD-IQ3_XXS  | 13.2 GB   | Acceptable| YES — best for speed |
| UD-Q3_K_S   | 15.4 GB   | Good      | YES — tight on RAM |
| UD-Q3_K_M   | 16.6 GB   | Very Good | MAYBE — 16GB RAM limit |

**Recommendation: UD-IQ3_XXS** (13.2 GB)
- Used in the original thread with great results
- Leaves enough headroom in 16GB RAM for system + KV cache
- Unsloth Dynamic 2.0 preserves quality better than standard IQ3

## 3. KV Cache Quantization

### Q4_0 KV Cache (Recommended for Qwen3.6)
Using `--cache-type-k q4_0 --cache-type-v q4_0`:
- **Lossless on Qwen3.6** because only 10/40 layers use KV-cached attention
- The other 30 layers use Gated DeltaNet (linear attention) -- no KV cache needed
- Quarters KV cache memory vs FP16, halves vs q8_0
- Enables 128K context on 8GB VRAM
- Verified: identical outputs at temp=0.0 across code, math, factual, and logic tests

### q8_0 KV Cache (Fallback)
Using `--cache-type-k q8_0 --cache-type-v q8_0`:
- Halves KV cache memory vs FP16
- Enables 64K context in available memory
- Slightly faster (~3%) due to less dequantization overhead
- Use if 64K context is sufficient

### Build Requirement for Q4_0
Must build llama.cpp with `-DGGML_CUDA_FA_ALL_QUANTS=ON` to enable flash attention
CUDA kernels for Q4_0. Without this, Q4_0 falls back to slower non-FA paths
and loses ~5% generation speed.

### TurboQuant (experimental, not yet in mainline):
- `--cache-type-k turbo3 --cache-type-v turbo3` -- 4.9x compression
- Would enable 256K context but requires special llama.cpp build
- Not recommended for initial setup

## 4. Flash Attention

`--flash-attn on` is mandatory:
- ~30% VRAM reduction for attention computation
- Required for quantized KV cache types
- Syntax must include "on" explicitly

## 5. Memory Management

### --no-mmap
- Forces model fully into RAM at load time (slower startup, faster inference)
- Avoids page fault overhead during generation
- Reported 3 tok/s improvement on constrained systems
- Use with caution on 16GB RAM — may cause swap pressure during load

### --mlock
- Prevents OS from swapping model pages to disk
- Requires `CAP_IPC_LOCK` or `ulimit -l unlimited`
- Critical for stable long-running inference

## 6. Thread Configuration

The i7-14700HX has:
- 8 P-cores (16 threads with HT)
- 12 E-cores (12 threads)
- Total: 28 threads

For llama.cpp expert processing on CPU:
- Use `-t 16` (P-core threads only) for best per-thread performance
- E-cores have lower IPC and can hurt MoE expert throughput
- Alternative: `-t 20` if E-cores help with batch processing

## 7. Batch Size

- `-b 2048 -ub 512` (defaults) — adequate for single-user
- Higher values increase prompt processing speed but use more VRAM
- For 8GB VRAM, keep defaults to avoid compute buffer VRAM pressure

## 8. Engine Choice: llama.cpp vs ik_llama.cpp

### ik_llama.cpp advantages:
- ~1.9x faster MoE inference (fused expert operations)
- Better GPU offload threshold for sparse experts
- Quantized matmul CUDA kernels for all quant types
- Graph reuse (`-gr`) reduces kernel launch overhead

### ik_llama.cpp drawbacks:
- Less tested, more niche
- CUDA-only focus (fine for us)
- May have compatibility issues with newest models

**Recommendation: Start with mainline llama.cpp, benchmark, then try ik_llama.cpp**

## 9. What NOT to Do

- Do NOT use CUDA 13.2 (produces gibberish with Qwen3.6)
- Do NOT use speculative decoding (drops throughput on MoE models)
- Do NOT use DFlash (not stable for MoE in llama.cpp yet)
- Do NOT set ncmoe too high (>30) — wastes GPU by underutilizing VRAM
- Do NOT set ncmoe too low (<21) — causes VRAM overflow and thrashing

## 10. Expected Performance

Based on research extrapolation for RTX 4070 Max-Q 8GB + 16GB RAM:
- **Conservative (ncmoe 25):** 40-45 tok/s
- **Optimal (ncmoe 23):** 43-48 tok/s
- **Aggressive (ncmoe 21):** Risk of OOM, may drop to 20 tok/s

The RTX 4070 has 256 GB/s memory bandwidth vs 3070 Ti's 192 GB/s,
giving us a ~33% bandwidth advantage for GPU-resident operations.

## Sources

- [MoE Offload Guide (HuggingFace)](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [DocShotgun's MoE Optimization Gist](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)
- [Qwen3.6 on 24GB (Amine Raji)](https://aminrj.com/posts/llamacpp-qwen36-35b/)
- [Best Way to Run Qwen 3.6 35B MoE Locally (InsiderLLM)](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/)
- [Unsloth Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- [TurboQuant Discussion](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [NVIDIA RTX llama.cpp Blog](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/)
