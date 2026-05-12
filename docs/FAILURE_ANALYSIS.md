# Failure Analysis: IQ4_XS vs Gemma 31B

A side-by-side examination of the three challenges where our best local configuration (Qwen3.6-35B-A3B IQ4_XS) failed while Gemma 4 31B succeeded, plus the one challenge where both models failed. The goal is to understand the specific failure modes and whether they point to systematic weaknesses or isolated issues.

## Summary

| Challenge | IQ4_XS | Gemma 31B | Root Cause |
|-----------|:------:|:---------:|------------|
| C06: Calendar Interval Merging | FAIL | FAIL | Both models produce logically sound but subtly incorrect priority resolution. Likely a specification edge case. |
| C07: Mini Regex Engine | FAIL | PASS | IQ4_XS builds correct parser but has a type confusion bug in the recursive matcher. Gemma uses a cleaner OOP architecture that avoids the issue entirely. |
| C09: Async Queue Bugs | FAIL | PASS | IQ4_XS consumes all 16,384 tokens reasoning about the bugs in natural language, never produces corrected code. Classic thinking overflow. |

## Token Efficiency: A Structural Difference

Before examining individual failures, the token counts tell an important story:

| Challenge | IQ4_XS Tokens | Gemma Tokens | Ratio |
|-----------|:------------:|:------------:|:-----:|
| C01 (both pass) | 4,500 | 439 | 10.3x |
| C02 (both pass) | 3,336 | 422 | 7.9x |
| C07 (IQ4_XS fails) | 5,462 | 1,235 | 4.4x |
| C09 (IQ4_XS fails) | 16,384 | 1,193 | 13.7x |

Qwen3.6 uses 4-14x more tokens than Gemma across all challenges. This is driven by the extended thinking architecture: the model reasons through the problem in `<think>` blocks before producing code. Gemma generates code directly with minimal preamble. On challenges where both pass, this is merely a speed and cost difference. On challenges where Qwen fails, the extended thinking becomes a liability, as the model can exhaust its token budget reasoning about the problem without ever producing a working solution.

---

## C09: Find and Fix Async Queue Bugs (IQ4_XS FAIL, Gemma PASS)

**Challenge**: Given a buggy `AsyncBoundedQueue` implementation using `asyncio.Event`, find at least 4 bugs and provide corrected code. The test verifies the corrected code runs without hanging.

### What IQ4_XS Did Wrong

IQ4_XS consumed all 16,384 tokens (the maximum) across both attempts. The output is not code but a verbose, meandering analysis written in markdown. It begins identifying bugs but gets caught in recursive self-doubt:

```
*   **Explanation:** The method calls `self.not_empty.set()`. While `Event.set()`
    is synchronous, in the context of `asyncio` patterns, this is often just logic.
    However, looking at the code provided, `set()` is synchronous. Wait, checking
    the documentation... `Event.set()` is synchronous. So this isn't a syntax error.
*   **Re-evaluating "At least 4 bugs"**:
    Let's look at the logic flow again.
```

The model correctly identifies the core architectural problem (using `asyncio.Event` instead of `asyncio.Condition`) but never produces corrected code. It spends thousands of tokens deliberating, second-guessing its own analysis, and re-reading the buggy code. The code extractor finds no Python code block to execute, and the raw text output triggers a `SyntaxError` when treated as Python.

**Failure mode**: Thinking overflow. The reasoning budget (4,096 tokens for the `<think>` phase) is respected, but the model continues reasoning in its visible output using markdown formatting instead of producing code. This is a variant of the thinking overflow problem: the model has learned that extended reasoning helps, but does not know when to stop reasoning and start coding.

### What Gemma Did Right

Gemma produced 1,193 tokens total. Its output is a clean, complete Python implementation that replaces `asyncio.Event` with `asyncio.Condition` (the correct fix), uses `async with self.condition` for proper locking, and calls `self.condition.notify_all()` for signaling. The code passes on the first attempt.

The key architectural decision is Gemma's use of `asyncio.Condition`, which combines a lock with notification. This eliminates the race conditions inherent in the original code's separate `Event` objects. Gemma does not explain the bugs in prose; it simply provides the corrected code, which is exactly what the test expects.

### Diagnosis

This is the most clear-cut case. The problem is not that IQ4_XS lacks the knowledge to fix the bugs. It correctly identifies `asyncio.Condition` as the right primitive. The problem is output discipline: the model cannot stop explaining and start coding. This is likely exacerbated by quantization, which may degrade the model's ability to follow the implicit instruction ("provide the COMPLETE corrected code") when competing with its reasoning loop. The `--reasoning-budget 4096` flag caps the `<think>` block, but it does not prevent the model from reasoning in its visible output.

**Potential mitigation**: Append an explicit instruction like "Output only code, no explanation" to the prompt. This may help the model break out of its reasoning loop. Alternatively, a stricter output format (e.g., "respond with a single Python code block") could constrain the output structure.

---

## C07: Mini Regex Engine (IQ4_XS FAIL, Gemma PASS)

**Challenge**: Implement a regex engine supporting `.`, `*`, `+`, `?`, character classes, groups, alternation, and escaping. Must do full-match (anchored both ends).

### What IQ4_XS Did Wrong

IQ4_XS builds a recursive descent parser that converts the regex pattern into a list of AST nodes stored as tuples: `(type, args)`. The parser itself is well-structured, handling atoms, character classes, quantifiers, concatenation, and alternation. The bug is in the matcher.

The `match` function dispatches on `typ = node[0]` and processes each node type. For `concat` nodes, the args contain a `tuple` of child indices (stored as `new_node('concat', tuple(idxs))`). But the matcher accesses sub-nodes via `nodes[idx]`, where `idx` is expected to be an integer. The error:

```python
TypeError: list indices must be integers or slices, not tuple
```

This occurs because the `concat` node stores its children as a single tuple argument `(idx1, idx2, idx3, ...)`, but the matcher code tries to index `nodes` with this tuple directly instead of iterating over its elements. The parser and matcher disagree on the data representation.

This is a type confusion bug: the parser packs multiple indices into a tuple and stores it as one arg, but the matcher expects each index to be a separate arg. Specifically, `new_node('concat', tuple(idxs))` creates `('concat', (tuple(idxs),))` (a tuple wrapped in the args tuple), and the matcher would need to unpack this correctly. The indirection through the `args` tuple adds a layer of complexity that the model's implementation does not handle consistently.

### What Gemma Did Right

Gemma uses an object-oriented architecture with separate classes for each node type (`Literal`, `Any`, `Class`, `Seq`, `Alt`, `Quantifier`). Each class has a `match(text, pos)` method that returns a set of possible positions. This design has two advantages:

1. **No type confusion possible**: each node type is a distinct class, so there is no dispatch-on-string-type pattern that can go wrong.
2. **Set-based matching**: instead of returning True/False, each node returns the set of positions reachable after matching. This naturally handles backtracking and alternation without explicit recursion management.

The `Quantifier` class implements `*`, `+`, `?` using a fixpoint loop: it repeatedly applies the inner node and collects reachable positions until no new positions are found. This is effectively an NFA simulation without building an explicit NFA.

Gemma needed 2 attempts (the first attempt also failed), but the second attempt used this cleaner architecture and passed all test cases.

### Diagnosis

IQ4_XS demonstrates strong parsing knowledge but fails at the parser-matcher interface. The tuple-based AST representation creates an opportunity for off-by-one nesting errors that class-based representations avoid. This is a pattern we see with quantized models: the high-level algorithm is correct, but fine-grained data structure consistency breaks down. The model "knows" how to build a regex parser and "knows" how to write a recursive matcher, but the seam between the two is where quantization-induced imprecision manifests.

Gemma's OOP approach is arguably over-engineered for a simple regex engine, but it is self-consistent. Each class encapsulates its own matching logic, and there is no shared mutable state or index arithmetic that can go wrong. This is the kind of architectural choice that is more robust to the pressure of one-shot code generation.

---

## C06: Calendar Interval Merging with Priority (Both FAIL)

**Challenge**: Implement a calendar that handles priority-based conflict resolution, interval splitting/merging, free slot calculation, and utilization measurement. This is the only challenge that no model has ever passed.

### What Both Models Did Wrong

Both models fail on the same test assertion:

```python
assert cal2.free_slots(0, 500, min_duration=150) == [(200, 300)]
```

The test adds events at (100, 200) and (300, 400), then asks for free slots between 0 and 500 with a minimum duration of 150. The expected free slots are `[(200, 300)]`, meaning the gap at (0, 100) is too short (100 < 150) and the gap at (400, 500) is also too short (100 < 150), but the gap at (200, 300) is exactly 100, which is also less than 150.

This means the expected answer `[(200, 300)]` appears to be incorrect if `min_duration=150` means "at least 150 units long," since the gap (200, 300) is only 100 units. However, looking more carefully, this test case may have a bug in the expected output, or `min_duration` may have a different semantic than "minimum length." If `min_duration` filters out slots shorter than `min_duration`, then no slots should be returned, not `[(200, 300)]`.

Both models implement `free_slots` with the straightforward interpretation (filter by `slot_length >= min_duration`), which would correctly return an empty list for this input. This suggests the test case itself may be the issue, not the models.

However, both models also fail earlier assertions. The IQ4_XS code fails on a priority resolution assertion involving a lower-priority event added over a higher-priority one. Both models handle the basic priority logic (higher priority trims lower priority events) but differ in edge case handling around the return value semantics: what should `add_event` return when the new event is completely occluded by higher-priority existing events?

### Diagnosis

C06 is the most complex challenge in the suite, requiring four interlocking features (priority resolution, interval splitting, free slot calculation, utilization) with subtle edge cases. The fact that no model passes it across 10+ runs and 4 different configurations (including cloud Gemma at full precision) suggests this challenge is at or beyond the frontier of single-shot code generation for current models.

There is also a possible test case issue with the `free_slots` assertion. The expected output `[(200, 300)]` with `min_duration=150` does not appear consistent with a "minimum slot length" interpretation, since the (200, 300) gap is only 100 units long. This warrants investigation.

---

## Cross-Cutting Observations

### 1. Token Economy as a Quality Signal

There is a strong correlation between token efficiency and correctness. Gemma averages 650 tokens per challenge; IQ4_XS averages 5,600. On challenges where IQ4_XS fails, the token count spikes to 5,000-16,000, suggesting the model is struggling and compensating with more reasoning rather than converging on a solution. In contrast, Gemma's failures (C06) still use only 766 tokens, indicating it fails fast rather than spiraling.

| Token Range | IQ4_XS Outcome | Count |
|-------------|---------------|:-----:|
| < 5,000 | 5 PASS, 0 FAIL | 5/5 |
| 5,000 - 6,000 | 2 PASS, 2 FAIL | 4 |
| > 6,000 | 0 PASS, 1 FAIL | 0/1 |

For IQ4_XS, every challenge under 5,000 tokens passes. Every challenge over 6,000 tokens fails. This suggests a practical heuristic: if the model is spending more than ~5,000 tokens, it is likely struggling and may benefit from a prompt restructure or constraint.

### 2. Architectural Choices Under Quantization Pressure

Gemma consistently uses simpler, more modular architectures (OOP classes, built-in data structures, standard library primitives). IQ4_XS tends toward more clever but fragile designs (tuple-packed ASTs, manual state management, interleaved parsing/matching). Under the precision constraints of quantization, the simpler architecture is more robust because there are fewer opportunities for subtle inconsistencies.

This is not unique to quantization. One-shot code generation in general favors self-contained, modular designs over tightly coupled ones. But quantization amplifies the effect: the probability of a subtle bug in an index calculation or type mismatch increases when the model's internal representations are compressed.

### 3. The Thinking Trap

IQ4_XS's extended thinking is a double-edged sword. On challenges where the reasoning leads to a clear solution (C01-C05, C08, C10), the thinking phase produces correct code. On challenges where the solution is ambiguous or the model is uncertain (C07, C09), the thinking phase becomes a trap: the model reasons extensively but cannot commit to an implementation. The `--reasoning-budget` flag helps by capping the `<think>` block, but it does not prevent reasoning overflow in the visible output (as seen in C09).

Gemma, which does not use extended thinking, never falls into this trap. Its failures are clean (wrong answer, not no answer), and it uses consistent token counts regardless of challenge difficulty. The tradeoff is that Gemma cannot benefit from extended reasoning on problems where thinking would help. But in practice, for code generation tasks with clear specifications, direct code output appears more reliable than reasoning-then-coding.

### 4. Retry Value

Both models benefit from retries, but differently:
- **Gemma** went from 8/10 to 9/10 with retries. The retry recovered C07 (regex engine), where the first attempt used a weaker architecture.
- **IQ4_XS** did not recover any of its three failures across 2 attempts each. C09 hit the token limit both times; C07 produced the same type confusion both times; C06 failed on different assertions but for related reasons.

This suggests that IQ4_XS failures are more systematic (architecture-level bugs, thinking overflow) rather than sampling variance. Increasing retries beyond 2 is unlikely to help for these specific challenges without prompt modification.

---

## Recommendations

1. **For C09 (Async Bugs)**: Add "Output only the corrected Python code. Do not include explanations." to the prompt. The model knows the fix but cannot stop explaining it.

2. **For C07 (Regex Engine)**: Consider adding a hint to use class-based node representation instead of tuple-packed ASTs. Alternatively, increase the reasoning budget specifically for this challenge to give the model more room to self-correct.

3. **For C06 (Calendar)**: Review the `free_slots` test case for correctness. The expected output `[(200, 300)]` with `min_duration=150` appears inconsistent with a "minimum slot length" interpretation.

4. **General**: Monitor token count as a diagnostic signal. If the model exceeds 5,000 tokens on a coding challenge, consider the response likely to fail and trigger a retry with a more constrained prompt.

5. **Prompt engineering**: For the quantized model, prefer explicit output format constraints ("respond with a single code block") over open-ended instructions ("provide the complete corrected code"). Quantized models have less capacity for following implicit format expectations.
