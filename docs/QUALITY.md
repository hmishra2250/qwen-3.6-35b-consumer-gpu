# Quality Verification: Q4_0 vs q8_0 KV Cache

## Why Q4_0 Works on Qwen3.6

Qwen3.6-35B-A3B uses **hybrid attention**:
- 30/40 layers: Gated DeltaNet (linear attention, no KV cache)
- 10/40 layers: Standard multi-head attention (uses KV cache)

Q4_0 KV quantization only affects 10 layers instead of all 40, reducing the impact by ~75% compared to a standard transformer. The recurrent DeltaNet layers act as a buffer, maintaining representation quality even with aggressive KV quantization.

## Test Methodology

### Tier 1: Perplexity + KL Divergence (Rigorous)

Uses `llama-perplexity` with wikitext-2 to measure:
- **Perplexity delta**: Direct quality metric (should be < 0.5)
- **KL divergence**: Statistical distance between output distributions (should be < 0.05)
- **Same-top-p %**: How often the top prediction is identical (should be > 95%)

Run: `./tests/perplexity.sh`

### Tier 2: Passkey Retrieval (Long-Context Stress)

Uses `llama-passkey` to hide a number in filler text and test retrieval. Directly stresses KV cache integrity at various context lengths.

Run: `./tests/passkey.sh`

### Tier 3: Hard Reasoning Tasks

Five challenging tasks at temperature=0.0:
1. **Multi-step calculus** (optimization with constraints)
2. **Einstein's riddle** (15-constraint logic puzzle)
3. **Thread-safe LRU cache** (concurrent code generation)
4. **Async bug detection** (finding 3+ bugs in rate limiter code)
5. **Physics derivation** (Schwarzschild radius from first principles)

Run: `./tests/quality.sh q4_0` and `./tests/quality.sh q8_0`

### Tier 4: Simple Verification (Quick Smoke Test)

Run: `./test.sh`

## Results: Simple Tests (temperature=0.0)

| Test | Q4_0 KV (128K ctx) | q8_0 KV (64K ctx) | Match? |
|------|-------------------|-------------------|--------|
| Fibonacci code | Correct, type hints, memoization | Identical logic | Yes |
| Math: 17*23+45-12*3 | 400 (correct steps) | 400 (identical steps) | Yes |
| Planets by mass | Jupiter, Saturn, Neptune, Uranus, Earth | Identical | Yes |
| Syllogism validity | Invalid (undistributed middle) | Invalid (same reasoning) | Yes |

## Results: Hard Tests (temperature=0.0, 2026-05-11)

Tested on RTX 4070 Max-Q 8GB. Q4_0 at 128K context, q8_0 at 64K context.

| Test | Q4_0 Answer | q8_0 Answer | Correct? | Match? |
|------|------------|------------|----------|--------|
| Multi-step calculus | x=50,y=25 (Part1), x=25,y=25 (Part2) | Same values | Yes | Same conclusions |
| Einstein's riddle | German owns the fish | German owns the fish | Yes | Same answer |
| Thread-safe LRU cache | DLL + dict + Lock + stress test | DLL + dict + Lock + stress test | Yes | Same architecture |
| Async bug detection | (16K tokens all in thinking) | (16K tokens all in thinking) | N/A | Both overflowed |
| Schwarzschild radius | R_s = 2GM/c^2, Earth = 8.87mm | R_s = 2GM/c^2, Earth = 8.87mm | Yes | Same derivation |

### Key Observations

1. **All answers are mathematically identical** - same formulas, same numerical results
2. **Phrasing differs** - Q4_0 and q8_0 use different wording/structure, but this is expected with temperature=0.0 on quantized models (the thinking tokens diverge, leading to different but equivalent phrasings)
3. **No degradation detected** - no wrong answers, no logic contradictions, no missing steps
4. **Both configs overflow on bug detection** - this is a model behavior issue (too much thinking), not a quantization issue

Raw results: `tests/results/hard_results_q4_0.json` and `tests/results/hard_results_q8_0.json`

Re-run: `./tests/quality.sh q4_0` and `./tests/quality.sh q8_0`

## What To Watch For

Signs of KV cache degradation:
- **Perplexity increase > 0.5**: Measurable quality loss
- **Passkey failure at lower junk levels**: KV cache losing information
- **Wrong math answers**: Accumulated precision errors in attention
- **Logic contradictions**: Model contradicting its own earlier reasoning
- **Code bugs introduced**: Subtle errors in generated code that q8_0 doesn't have

## Decision Criteria

| Metric | Acceptable | Warning | Reject Q4_0 |
|--------|-----------|---------|-------------|
| PPL delta | < 0.2 | 0.2-0.5 | > 0.5 |
| KL divergence | < 0.02 | 0.02-0.05 | > 0.05 |
| Passkey accuracy | Same as q8_0 | -5% | -10% |
| Hard test answers | Identical | Minor phrasing diff | Wrong answers |
