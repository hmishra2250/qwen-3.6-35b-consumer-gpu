# Project Goal: Qwen3.6-35B-A3B Maximum Performance on Limited Hardware

## System Specs
- GPU: NVIDIA GeForce RTX 4070 Max-Q (Laptop) -- 8GB VRAM
- CPU: Intel Core i7-14700HX -- 28 threads (8P + 6E cores)
- RAM: 16GB DDR5
- OS: Linux (Ubuntu) -- Kernel 6.17.0
- CUDA: 12.6

## Target Model
- Qwen3.6-35B-A3B (MoE: 35B total params, ~3B active per token)
- 256 routed experts, 8 active per forward pass
- Hybrid attention: 30/40 Gated DeltaNet + 10/40 standard MHA
- Native context: 262,144 tokens

## Achieved Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Context window | 64K | **128K** |
| Generation speed | 35-45 tok/s | 10-15 tok/s (IQ4_XS) / 43 tok/s (IQ3_XXS) |
| Quality | No degradation | **9/10 SWE challenges** (1 point from cloud Gemma 31B) |
| VRAM usage | 6.5-7.5 GB | 5.7 GB (IQ4_XS) / 7.2 GB (IQ3_XXS) |
| RAM usage | 12-13 GB | ~14.5 GB (IQ4_XS) / ~13 GB (IQ3_XXS) |
| API compatibility | OpenAI-compatible | Yes, via llama-server |

Speed is lower with IQ4_XS due to more expert offloading (ncmoe=30 vs 25), but the quality improvement (9/10 vs 5/10) makes it the clear winner for tasks where correctness matters.

## Strategy
The key insight was exploiting two properties of Qwen3.6's hybrid architecture:

1. **MoE sparsity** (only 3B of 35B params active per token): offload most expert weights to CPU via `--n-cpu-moe`, keeping attention layers on GPU

2. **Hybrid attention** (only 10/40 layers use KV cache): aggressive KV quantization has ~75% less impact than on standard transformers

Combined with IQ4_XS weights (crossing the 4-bit threshold), asymmetric KV cache, and `/no_think` mode for code generation, a $1,500 laptop matches cloud-served Gemma 31B within 1 point.

## Success Criteria (All Met)

1. Stable generation at 128K context without OOM -- **achieved**
2. No OOM crashes under sustained load -- **achieved** (ncmoe=30 gives 2.5 GB VRAM headroom)
3. Quality competitive with cloud models -- **achieved** (9/10 vs Gemma's 10/10)
4. Server accessible via OpenAI-compatible API on localhost -- **achieved**
