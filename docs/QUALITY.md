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

## Results: Hard Tests (temperature=0.0)

*Run `./tests/quality.sh q4_0` and `./tests/quality.sh q8_0` to generate fresh results.*

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
