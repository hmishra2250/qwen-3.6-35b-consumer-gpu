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

**Sweet spot depends on quantization:**
- IQ3_XXS (13.2 GB): ncmoe=25 (43 tok/s, 1 GB VRAM headroom)
- IQ4_XS (17.7 GB): ncmoe=30 (10-15 tok/s, ~2.5 GB VRAM headroom, RAM is the constraint)

## 2. Quantization Choice

### Two options for 8GB VRAM + 16GB RAM:

| Quant | File Size | SWE Score | Speed | Recommendation |
|-------|-----------|:---------:|-------|----------------|
| **UD-IQ4_XS** | 17.7 GB | **9/10** | ~10-15 tok/s | Best quality, recommended |
| UD-IQ3_XXS | 13.2 GB | 5/10 | ~43 tok/s | Best speed, lower quality |

**IQ4_XS is recommended.** It crosses the 4-bit reliability threshold, scoring 9/10 on hard SWE challenges vs 5/10 for IQ3_XXS. The speed tradeoff (~3x slower) is acceptable for tasks where correctness matters.

IQ3_XXS sits below the 4-bit threshold. Research consistently shows that sub-4-bit quantization degrades tool calling, complex reasoning, and multi-step code generation. The quality gap is not a minor precision loss but a qualitative shift in capability.

## 3. KV Cache Quantization

### Asymmetric KV Cache (Recommended)
Using `--cache-type-k q4_0 --cache-type-v q8_0`:
- **Near-lossless on Qwen3.6** because only 10/40 layers use KV-cached attention
- The other 30 layers use Gated DeltaNet (linear attention) -- no KV cache needed
- The value cache is ~3.5x more sensitive to quantization than the key cache (QAQ paper)
- Asymmetric quantization protects values while saving VRAM on keys
- Enables 128K context on 8GB VRAM
- KV cache errors accumulate at very long contexts (PM-KVQ paper); asymmetric KV mitigates this

### Symmetric Q4_0 (Fallback)
Using `--cache-type-k q4_0 --cache-type-v q4_0`:
- Maximum VRAM savings
- Near-lossless at short contexts; quality may degrade at very long contexts
- Adequate for most use cases

### q8_0 KV Cache (Conservative)
Using `--cache-type-k q8_0 --cache-type-v q8_0`:
- Maximum KV quality, enables 64K context
- Slightly faster (~3%) due to less dequantization overhead

### Build Requirement
Must build llama.cpp with `-DGGML_CUDA_FA_ALL_QUANTS=ON` to enable flash attention CUDA kernels for quantized KV. Without this, quantized KV falls back to slower non-FA paths and loses ~5% speed.

## 4. Inference Settings (Critical for Qwen3.6)

Three settings are mandatory. Getting any wrong causes severe degradation:

| Setting | Value | What Happens Without It |
|---------|-------|------------------------|
| `temperature=0.6` | Per-request | Repetition loops, stuck generation. Official docs warn against temp=0. |
| `--reasoning-budget 4096` | Server flag | Model spends all tokens in `<think>` blocks, produces no visible answer. |
| `--jinja` + `preserve_thinking` | Server flag | Model re-reasons from scratch each turn, wasting tokens. |
| `--reasoning-budget-message` | Server flag | Hard cutoff (78% HumanEval) instead of graceful transition (89%). |
| `/no_think` in prompt | For code tasks | Model reasons excessively in visible output instead of producing code. |

### /no_think Mode
Appending `/no_think` to the user message disables the thinking phase. For code generation:
- 9/10 SWE score (vs 7/10 with thinking)
- Eliminates thinking overflow (model spending all tokens reasoning, never coding)
- Forces simpler, more modular architectures that are robust under quantization
- Only ~10-20% of tasks genuinely need thinking (complex recursive logic, state machines)

### preserve_thinking
`--chat-template-kwargs '{"preserve_thinking":true}'` retains the model's reasoning in conversation history. Without it, the model re-derives everything from scratch each turn. Community reports identify this as the single most impactful fix for Qwen3.6 reasoning loops.

### reasoning-budget-message
`--reasoning-budget-message "I need to provide my answer now."` provides a graceful transition when the thinking budget is exhausted, instead of a hard cutoff. Recovers 89% vs 78% on HumanEval benchmarks.

## 5. Flash Attention

`--flash-attn on` is mandatory:
- ~30% VRAM reduction for attention computation
- Required for quantized KV cache types
- Syntax must include "on" explicitly

## 6. Memory Management

### --no-mmap
- Forces model fully into RAM at load time (slower startup, faster inference)
- Avoids page fault overhead during generation
- Reported 3 tok/s improvement on constrained systems

### --mlock
- Prevents OS from swapping model pages to disk
- Requires `CAP_IPC_LOCK` or `ulimit -l unlimited`
- Critical for stable long-running inference

## 7. Thread Configuration

The i7-14700HX has:
- 8 P-cores (16 threads with HT)
- 12 E-cores (12 threads)
- Total: 28 threads

For llama.cpp expert processing on CPU:
- Use `-t 16` (P-core threads only) for best per-thread performance
- E-cores have lower IPC and can hurt MoE expert throughput

## 8. What NOT to Do

- Do NOT use CUDA 13.2 (produces gibberish with Qwen3.6)
- Do NOT use speculative decoding (drops throughput on MoE models)
- Do NOT set temperature=0.0 (causes repetition loops, score drops from 7/10 to 6/10)
- Do NOT omit reasoning-budget (model spends all tokens thinking, produces no answer)
- Do NOT set ncmoe too high (>32) for IQ4_XS -- wastes GPU by underutilizing VRAM
- Do NOT set ncmoe too low for your model -- causes OOM and thrashing

## Sources

- [MoE Offload Guide (HuggingFace)](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [DocShotgun's MoE Optimization Gist](https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0)
- [Qwen3.6 on 24GB (Amine Raji)](https://aminrj.com/posts/llamacpp-qwen36-35b/)
- [Best Way to Run Qwen 3.6 35B MoE Locally (InsiderLLM)](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/)
- [Unsloth Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- [QAQ: Quality Adaptive Quantization for LLM KV Cache](https://arxiv.org/abs/2403.04643)
- [PM-KVQ: Post-training Multi-bit KV Cache Quantization](https://arxiv.org/abs/2412.15307)
- [r/LocalLLaMA: Reasoning Budget Controls Analysis](https://insights.marvin-42.com/articles/rlocalllama-tracks-llamacpps-new-reasoning-budget-controls)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
