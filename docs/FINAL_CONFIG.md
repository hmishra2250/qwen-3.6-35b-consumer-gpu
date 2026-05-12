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
File:  Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
Size:  17.7 GB on disk
Quant: Unsloth Dynamic 2.0 IQ4_XS (~4-bit, intelligent layer upcasting)
Arch:  MoE -- 35B total params, ~3B active per token (8/256 experts)
       Hybrid attention: 30/40 layers = Gated DeltaNet (linear), 10/40 = standard KV
```

IQ4_XS crosses the 4-bit reliability threshold, scoring 9/10 on SWE challenges vs 5/10 for IQ3_XXS. The IQ3_XXS quantization (13.2 GB) is available as a fallback with smaller footprint but significantly lower quality.

## Optimal Server Command
```bash
./run-iq4xs.sh

# Or manually:
llama-server \
  -m Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  -ngl 99 \
  --n-cpu-moe 30 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q8_0 \
  --ctx-size 131072 \
  -np 1 \
  -t 16 \
  --no-mmap \
  --jinja \
  --reasoning-budget 4096 \
  --reasoning-budget-message "I need to provide my answer now." \
  --reasoning-format deepseek \
  --chat-template-kwargs '{"preserve_thinking":true}' \
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

`GGML_CUDA_FA_ALL_QUANTS=ON` enables flash attention CUDA kernels for quantized KV cache types. Without this flag, quantized KV falls back to slower non-FA paths. This is the single most impactful build flag.

## Flag Justification

| Flag | Value | Why |
|------|-------|-----|
| `-ngl 99` | All layers to GPU | Maximizes GPU utilization; ncmoe then selectively moves experts back |
| `--n-cpu-moe 30` | 30 layers' experts on CPU | Fits IQ4_XS (17.7 GB) within 8 GB VRAM; keeps ~5.7 GB on GPU |
| `--flash-attn on` | Enabled | 30% VRAM savings on attention, required for quantized KV cache |
| `--cache-type-k q4_0` | 4-bit K cache | Keys are less sensitive to quantization (~3.5x less than values) |
| `--cache-type-v q8_0` | 8-bit V cache | Protects the more sensitive value cache while saving VRAM on keys |
| `--ctx-size 131072` | 128K tokens | Full context, fits with asymmetric KV savings |
| `--jinja` | Jinja2 templates | Required for preserve_thinking and enable_thinking controls |
| `--reasoning-budget 4096` | Max thinking tokens | Caps the `<think>` phase; without it, model can spend all tokens thinking |
| `--reasoning-budget-message` | Graceful transition | Recovers 89% vs 78% HumanEval compared to hard cutoff |
| `--reasoning-format deepseek` | DeepSeek format | Separates reasoning_content from visible content in API responses |
| `--chat-template-kwargs` | preserve_thinking | Retains prior reasoning in conversation history, prevents re-reasoning loops |
| `-np 1` | 1 parallel slot | Single user -- no memory wasted on multiple KV caches |
| `-t 16` | 16 threads | P-cores only (0-15). E-cores have lower IPC and hurt MoE throughput |
| `--no-mmap` | Disabled mmap | Avoids page fault overhead; preloads to RAM for stable inference |

## Prompt Strategy

For code generation tasks, append `/no_think` to the user message. This disables the thinking phase and produces direct code output, which scores 9/10 vs 7/10 with thinking enabled. Combine with a code-only system prompt:

```json
{"role": "system", "content": "You are a code generator. Respond with only the requested code in a single Python code block. No explanations."}
```

For complex reasoning tasks (recursive data structures, state machines, protocol design), omit `/no_think` to enable the thinking phase.

## Expected Performance

| Metric | Expected Range |
|--------|---------------|
| Token generation | 10-15 tok/s |
| VRAM usage | ~5.7 GB (model) + KV cache |
| RAM usage | ~14-15 GB |
| Context window | 128K tokens |
| SWE challenge score | 9/10 (/no_think), 7/10 (thinking) |

## Memory Budget Breakdown

```
Total VRAM: 8188 MiB
-----------------------
Model weights (on GPU):           ~4200 MiB  (non-expert layers, IQ4_XS)
KV cache (q4_0 K + q8_0 V, 128K): ~900 MiB  (10 attn layers only)
Recurrent state (DeltaNet):         ~63 MiB  (30 layers, float32)
Compute buffer:                    ~500 MiB
-----------------------
Total GPU:                        ~5663 MiB  (headroom: ~2500 MiB)

Total RAM: 16 GB
-----------------------
Expert weights (30 layers on CPU):~11000 MiB
Host compute buffer:                ~264 MiB
System + OS:                       ~3.0 GB
-----------------------
Total RAM:                         ~14.5 GB (~1.5 GB headroom)
```

Note: RAM is the tighter constraint with IQ4_XS. The larger model (17.7 GB vs 13.2 GB for IQ3_XXS) puts more weight on CPU. VRAM has comfortable headroom.

## Quality Assurance

- IQ4_XS crosses the 4-bit reliability threshold, scoring 9/10 on hard SWE challenges
- Asymmetric KV cache (Q4 keys + Q8 values) protects the more sensitive value cache
- KV cache quantization is near-lossless at short contexts on hybrid architectures (only 10/40 layers affected)
- KV cache errors do accumulate at very long contexts; the value cache is ~3.5x more sensitive than keys
- Verified: identical simple outputs at temp=0.0 across code, math, factual, and logic tests
- `/no_think` mode eliminates thinking overflow, the primary failure mode for code generation

## Fallback Configurations

### IQ3_XXS (smaller footprint, lower quality):
```bash
./run.sh    # IQ3_XXS, 128K context, symmetric Q4_0 KV, 5/10 SWE score
```

### High-speed mode (64K context):
```bash
KV_TYPE=q8_0 CTX=65536 ./run.sh    # Less context, more VRAM headroom
```

### If RAM pressure (swap thrashing):
```bash
CTX=65536 ./run-iq4xs.sh    # Halve context
```

### If thermal throttling (laptop):
Sustained long generations may throttle at 85C. The 1024-token sustained rate is the true benchmark.

## API Usage

Once server is running, it exposes an OpenAI-compatible API:

```bash
# Chat completion (with /no_think for code tasks)
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Implement quicksort in Python /no_think"}],
    "temperature": 0.6,
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
