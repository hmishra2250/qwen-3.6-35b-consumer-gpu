# Final Optimal Configuration

## Hardware Profile
```
GPU:  RTX 4070 Max-Q (8188 MiB VRAM, 256 GB/s bandwidth, Ada Lovelace sm_89)
CPU:  i7-14700HX (8P + 6E cores, 28 threads)
RAM:  16 GB DDR5
CUDA: 12.6 toolkit / 12.8 driver capability
```

## Chosen Model
```
File:  Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
Size:  13.2 GB on disk
Quant: Unsloth Dynamic 2.0 IQ3_XXS (3-bit, intelligent layer upcasting)
Arch:  MoE -- 35B total params, ~3B active per token (8/256 experts)
       Hybrid attention: 30/40 layers = Gated DeltaNet (linear), 10/40 = standard KV
```

## Optimal Server Command
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

## Required Build Flags
```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_COMPRESSION_MODE=speed
```

`GGML_CUDA_FA_ALL_QUANTS=ON` enables flash attention CUDA kernels for Q4_0 KV cache. Without this flag, Q4_0 falls back to slower non-FA paths. This is the single most impactful build flag.

## Flag Justification

| Flag | Value | Why |
|------|-------|-----|
| `-ngl 99` | All layers to GPU | Maximizes GPU utilization; ncmoe then selectively moves experts back |
| `--n-cpu-moe 25` | 25 layers' experts on CPU | Best speed/stability tradeoff: 43 tok/s with 1033 MiB headroom |
| `--flash-attn on` | Enabled | 30% VRAM savings on attention, required for quantized KV cache |
| `--cache-type-k q4_0` | 4-bit K cache | Lossless on Qwen3.6 hybrid arch (only 10/40 layers use KV cache) |
| `--cache-type-v q4_0` | 4-bit V cache | Halves KV memory vs q8_0, enables 128K context |
| `--ctx-size 131072` | 128K tokens | 2x previous config, fits with Q4_0 KV savings |
| `-np 1` | 1 parallel slot | Single user -- no memory wasted on multiple KV caches |
| `-t 16` | 16 threads | P-cores only (0-15). E-cores have lower IPC and hurt MoE throughput |
| `--no-mmap` | Disabled mmap | Avoids page fault overhead; preloads to RAM for stable inference |

## Expected Performance

| Metric | Expected Range |
|--------|---------------|
| Token generation | 40-45 tok/s |
| Prompt processing | 100-145 tok/s |
| VRAM usage | 7.1-7.2 GB |
| RAM usage | 12-13 GB |
| Time to first token (short prompt) | <500ms |
| Context window | 128K tokens |

## Memory Budget Breakdown

```
Total VRAM: 8188 MiB
-----------------------
Model weights (on GPU):           ~5640 MiB  (non-expert layers)
KV cache (q4_0, 128K ctx):         ~720 MiB  (10 attn layers only)
Recurrent state (DeltaNet):         ~63 MiB  (30 layers, float32)
Compute buffer:                    ~499 MiB
-----------------------
Total GPU:                        ~6922 MiB  (headroom: ~1033 MiB)

Total RAM: 16 GB
-----------------------
Expert weights (25 layers on CPU): ~6947 MiB
Host compute buffer:                ~264 MiB
System + OS:                       ~3.0 GB
-----------------------
Total RAM:                         ~13.0 GB (3 GB headroom)
```

## Quality Assurance

- Q4_0 KV cache is lossless on Qwen3.6 because only 10/40 layers use KV-cached attention
- Verified: identical outputs on code gen, math, factual, and logic tests at temp=0.0
- IQ3_XXS with Unsloth Dynamic 2.0 preserves reasoning capability
- Qwen3.6-35B-A3B scored 73.4% SWE-Bench -- quality preserved at 3-bit

## Fallback Configurations

### High-speed mode (64K context):
```bash
KV_TYPE=q8_0 CTX=65536 ./run.sh    # ~44-46 tok/s, less context
```

### If OOM:
```bash
NCMOE=27 ./run.sh    # More experts on CPU
```

### If RAM pressure (swap thrashing):
```bash
CTX=65536 ./run.sh    # Halve context, saves ~360 MiB VRAM
```

### If thermal throttling (laptop):
Sustained 2048+ token generations may throttle at 85C. The 1024-token sustained rate is the true benchmark.

## API Usage

Once server is running, it exposes an OpenAI-compatible API:

```bash
# Chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 2048
  }'

# Streaming
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "temperature": 0.6
  }'
```

Compatible with any OpenAI SDK client by pointing base_url to `http://127.0.0.1:8080/v1`.
