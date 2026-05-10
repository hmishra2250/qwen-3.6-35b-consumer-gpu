# Project Goal: Qwen3.6-35B-A3B Maximum Performance on Limited Hardware

## System Specs
- GPU: NVIDIA GeForce RTX 4070 Max-Q (Laptop) — 8GB VRAM
- CPU: Intel Core i7-14700HX — 28 threads (14 cores, 20 P+E cores)
- RAM: 16GB DDR5
- OS: Linux (Ubuntu) — Kernel 6.17.0
- CUDA: 12.8
- Driver: 570.211.01

## Target Model
- Qwen3.6-35B-A3B (MoE: 35B total params, ~3B active per token)
- 256 routed experts, 8 active per forward pass
- Native context: 262,144 tokens

## Performance Target
- **Goal: 35-45 tokens/second generation speed** with 64K context
- **Constraint: No quality/information loss** — quantization chosen to preserve reasoning
- Acceptable VRAM usage: 6.5-7.5 GB (leaving ~700MB headroom)
- RAM usage target: ~12-13 GB (leaving 3GB for system)

## Strategy
The RTX 4070 Max-Q with 8GB matches the thread's RTX 3070 Ti scenario almost exactly.
The RTX 4070 has higher memory bandwidth (256 GB/s vs 192 GB/s on 3070 Ti) and newer
Ada Lovelace architecture, so we should achieve equal or better results than the thread.

Key insight: The `-ncmoe` (--n-cpu-moe) flag offloads MoE expert weights from specific
layers to CPU RAM while keeping attention + shared experts on GPU. Since only 8/256
experts activate per token, the PCIe transfer overhead for activations is minimal.

## Success Criteria
1. Stable generation at 35+ tok/s with 64K context
2. No OOM crashes under sustained load
3. Quality matches reference (no gibberish, coherent reasoning)
4. Server accessible via OpenAI-compatible API on localhost
