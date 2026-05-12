# Tuning Guide: Finding Your Sweet Spot

## Choosing a Quantization

| Quant | Size | SWE Score | Speed | Best For |
|-------|------|:---------:|-------|----------|
| **IQ4_XS** | 17.7 GB | **9/10** | ~10-15 tok/s | Quality-sensitive tasks (code, reasoning) |
| IQ3_XXS | 13.2 GB | 5/10 | ~43 tok/s | Speed-sensitive tasks, tight RAM |

Use `./run-iq4xs.sh` for IQ4_XS (recommended) or `./run.sh` for IQ3_XXS.

## The ncmoe Curve

Performance follows an inverted-U curve as you decrease ncmoe. The sweet spot depends on model size:

- **IQ3_XXS** (13.2 GB): ncmoe=25 is optimal (~43 tok/s, 1 GB VRAM headroom)
- **IQ4_XS** (17.7 GB): ncmoe=30 is required (~10-15 tok/s, RAM becomes the constraint)

```
Speed (tok/s) — IQ3_XXS
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

IQ4_XS operates further left on this curve (ncmoe=30) because its larger weights need more CPU offloading.

## Step-by-Step Tuning Process

### 1. Find your VRAM ceiling

```bash
# Check baseline VRAM usage (before loading model)
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Subtract from 8188 MiB total → your budget
```

### 2. Start with the recommended config

```bash
./run-iq4xs.sh    # IQ4_XS, ncmoe=30, asymmetric KV, 128K context
```

### 3. Adjust if needed

- If OOM with IQ4_XS → increase ncmoe: `NCMOE=32 ./run-iq4xs.sh`
- If RAM is tight → reduce context: `CTX=65536 ./run-iq4xs.sh`
- If RAM is critically tight → switch to IQ3_XXS: `./run.sh`

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

With Q4_0 KV cache, 128K context fits comfortably on 8GB VRAM (720 MiB vs 1360 MiB for q8_0). This is near-lossless on Qwen3.6 because only 10/40 layers use KV-cached attention. For best quality, use asymmetric KV (`--cache-type-k q4_0 --cache-type-v q8_0`) to protect the more sensitive value cache.

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

## Prompt Tuning: /no_think vs Thinking

| Mode | SWE Score | Best For |
|------|:---------:|----------|
| `/no_think` (default) | 9/10 | Code generation, clear specs, most tasks |
| Thinking | 7/10 | Complex reasoning, recursive data structures, state machines |

Append `/no_think` to user messages for code tasks. Omit it for problems requiring multi-step architectural reasoning.

## Advanced: ik_llama.cpp

If you want to experiment with alternative inference engines:

```bash
cd ~/Dev/2026/qwen-3.6-35b
git clone https://github.com/ikawrakow/ik_llama.cpp.git
cd ik_llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Run with graph reuse (reduces kernel overhead)
./build/bin/llama-server \
    -m ../models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
    -ngl 99 \
    --n-cpu-moe 30 \
    --flash-attn on \
    --cache-type-k q4_0 \
    --cache-type-v q8_0 \
    --ctx-size 131072 \
    -np 1 \
    -t 16 \
    --no-mmap \
    -gr \
    --host 127.0.0.1 \
    --port 8080
```

The `-gr` (graph reuse) flag is ik_llama.cpp-specific and eliminates repeated
graph construction overhead during generation. Not tested with current setup.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Gibberish output | Wrong CUDA version | Verify nvcc --version shows 12.6 |
| <10 tok/s | ncmoe too high | Decrease ncmoe |
| OOM crash | ncmoe too low | Increase ncmoe by 2 |
| Slow first token | Cold KV cache | Normal; subsequent tokens will be fast |
| System freeze | RAM exhausted | Add swap or reduce ctx-size |
| Unstable speed | Thermal throttling | Check GPU temp: nvidia-smi -q -d TEMPERATURE |
