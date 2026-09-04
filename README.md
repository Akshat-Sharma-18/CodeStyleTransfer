# Code Style Transfer

Fine-tune a small code model to rewrite code between styles (verbose ↔ terse, idiom
normalization, and eventually author-fingerprint transfer) while preserving behavior.
Every rewrite is checked by actually running the original test suite against it —
correctness is a hard metric here, not a matter of taste.

See [SPEC.md](SPEC.md) for the full project spec.

## Status

Week 1, in progress:

- 5-problem corpus with test suites (`two_sum`, `fizzbuzz`, `is_palindrome`,
  `fibonacci`, `max_subarray`)
- Correctness runner (`src/runner/sandbox.py`): runs candidate code against a
  problem's pytest suite in a subprocess
- Three mechanical transforms: identifier renaming, docstring/comment
  stripping, `for range(...)` → `while` — all AST-based and behavior-preserving
- `src/transforms/generate_pairs.py` applies every transform to every
  problem, verifies each result through the runner, and writes only the
  ones that still pass to `data/pairs/mechanical_pairs.jsonl` — 15/15 pairs
  currently verified
- Repo self-tests for the transforms and runner (`tests/`)

Next: expand the problem corpus, add more transform types (expand/inline
helpers, ternary ↔ if/else), then start the LoRA fine-tune (Week 2).

## Layout

```
data/
  raw/        source solutions pulled in for the corpus
  pairs/      generated (style_a, style_b) training pairs
  eval/       held-out real pairs (Option B) + eval outputs
src/
  transforms/ mechanical style transforms (rename, comment strip, loop-idiom swap, ...)
  runner/     sandboxed compile+test runner (the correctness gate)
  model/      LoRA fine-tuning + inference over the code backbone
  eval/       functional-correctness, style-hit, content-preservation metrics
problems/     problem statements + test suites for the corpus
notebooks/    exploration
tests/        tests for this repo's own code (transforms, runner, eval)
```

## Plan

| Week | Goal |
|------|------|
| 1 | Corpus + test suites + correctness runner + mechanical transform generator |
| 2 | LoRA fine-tune, both-direction rewriting, functional-correctness eval |
| 3 | Style proxies + classifier, three-metric eval table, baselines |
| 4 | Real-pair eval, error analysis (reward-hacking failure modes), writeup |
| 5+ | Stretch: reject-sampling/RL toward correctness, or unpaired round-trip (Option C) |

## Hardware target

24GB RAM, RTX 5050 (~8GB VRAM), Ryzen AI 7 (8-core) — LoRA/QLoRA on a ~350M–1B
code model (e.g. CodeT5+), test execution parallelized across CPU cores.
