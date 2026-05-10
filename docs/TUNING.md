# Tuning Guide: Finding Your Sweet Spot

## The ncmoe Curve

Performance follows an inverted-U curve as you decrease ncmoe:

```
Speed (tok/s)
  50 |              *
  45 |           *     *
  40 |        *           *
  35 |     *                 
  30 |  *                      *
  20 |                            *  ← OOM cliff
  10 | *
     +--+--+--+--+--+--+--+--+--+--→ ncmoe
      35  33  30  27  25  23  21  19
       ← more on CPU          more on GPU →
```

## Step-by-Step Tuning Process

### 1. Find your VRAM ceiling

```bash
# Check baseline VRAM usage (before loading model)
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Subtract from 8188 MiB total → your budget
```

### 2. Run benchmarks

```bash
./benchmark.sh
```

### 3. Interpret results

- If ncmoe=23 works without OOM → use it (expected ~43 tok/s)
- If ncmoe=23 OOMs → fall back to ncmoe=25 (~40 tok/s)
- If ncmoe=25 OOMs → reduce context: CTX=32768 NCMOE=23 ./run.sh

### 4. Monitor during real usage

```bash
# In a separate terminal while server is running:
watch -n 1 nvidia-smi
```

VRAM usage grows with context. A prompt that fills 64K tokens will use more VRAM
than a short prompt. If you see VRAM hit 7800+ MiB, increase ncmoe by 2.

## Thread Count Tuning

The i7-14700HX has P-cores (fast) and E-cores (slow):
- P-cores: cores 0-7 (threads 0-15 with HT)
- E-cores: cores 8-13 (threads 16-27)

### Recommended settings:
- `-t 16` — Uses only P-core threads. Best per-thread performance.
- `-t 20` — Adds some E-cores. May help prompt processing.
- `-t 28` — All threads. Usually worse due to E-core contention.

### To pin to P-cores explicitly:
```bash
taskset -c 0-15 ./run.sh
```

## Context Size vs Speed

Larger context = more KV cache = less VRAM for compute buffers.

| Context | KV (q4_0) | KV (q8_0) | Notes |
|---------|-----------|-----------|-------|
| 8192    | ~45 MiB   | ~85 MiB   | Minimal, fast |
| 32768   | ~180 MiB  | ~340 MiB  | Good balance |
| 65536   | ~360 MiB  | ~680 MiB  | Max for q8_0 on 8GB |
| 131072  | ~720 MiB  | ~1360 MiB | Max for q4_0 on 8GB |

With Q4_0 KV cache, 128K context fits comfortably on 8GB VRAM (720 MiB vs 1360 MiB for q8_0). This is lossless on Qwen3.6 because only 10/40 layers use KV-cached attention.

## Build Flags (Critical for Q4_0 Performance)

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_COMPRESSION_MODE=speed
```

`GGML_CUDA_FA_ALL_QUANTS=ON` enables flash attention CUDA kernels for Q4_0 KV cache. Without it, Q4_0 loses ~5% generation speed.

## Advanced: ik_llama.cpp (Phase 2)

If you want ~1.5-2x more performance after baseline is working:

```bash
cd ~/Dev/2026/qwen-3.6-35b
git clone https://github.com/ikawrakow/ik_llama.cpp.git
cd ik_llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Run with graph reuse (reduces kernel overhead)
./build/bin/llama-server \
    -m ../models/UD-IQ3_XXS/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
    -ngl 99 \
    --n-cpu-moe 23 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --ctx-size 65536 \
    -np 1 \
    -t 16 \
    --no-mmap \
    -gr \
    --host 127.0.0.1 \
    --port 8080
```

The `-gr` (graph reuse) flag is ik_llama.cpp-specific and eliminates repeated
graph construction overhead during generation.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Gibberish output | Wrong CUDA version | Verify nvcc --version shows 12.6 |
| <10 tok/s | ncmoe too high | Decrease ncmoe |
| OOM crash | ncmoe too low | Increase ncmoe by 2 |
| Slow first token | Cold KV cache | Normal; subsequent tokens will be fast |
| System freeze | RAM exhausted | Add swap or reduce ctx-size |
| Unstable speed | Thermal throttling | Check GPU temp: nvidia-smi -q -d TEMPERATURE |
