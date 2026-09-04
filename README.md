# Code Style Transfer

Fine-tune a small code model to rewrite code between styles (verbose ↔ terse, idiom
normalization, and eventually author-fingerprint transfer) while preserving behavior.
Every rewrite is checked by actually running the original test suite against it —
correctness is a hard metric here, not a matter of taste.

See [SPEC.md](SPEC.md) for the full project spec.

## Status

Week 1, in progress:

- 10-problem corpus with test suites (`two_sum`, `fizzbuzz`, `is_palindrome`,
  `fibonacci`, `max_subarray`, `binary_search`, `valid_parentheses`,
  `merge_intervals`, `reverse_words`, `count_vowels`)
- Correctness runner (`src/runner/sandbox.py`): runs candidate code against a
  problem's pytest suite in a subprocess
- Four mechanical transforms, all AST-based and behavior-preserving:
  identifier renaming, docstring/comment stripping, `for range(...)` →
  `while`, and `if/else` assignment → ternary
- `src/transforms/generate_pairs.py` applies every transform to every
  problem, skips no-ops (where a transform's pattern doesn't match),
  verifies the rest through the runner, and writes only the ones that
  still pass to `data/pairs/mechanical_pairs.jsonl` — 22 pairs currently
  verified
- Repo self-tests for the transforms and runner (`tests/`)

Week 3 work brought forward (the eval harness, built before training so that
training results can be trusted):

- Three-metric eval table (`src/eval/evaluate.py`): functional correctness,
  style-target hit, content preservation — scoring any rewriter passed in as
  a plain `(code, direction) -> code` callable, so every baseline and model
  goes through identical scoring
- Style proxies (`src/eval/style_proxies.py`): seven proxies, with the
  "moved toward terse" verdict requiring a majority of them, so no single
  gameable number decides the style metric
- Content preservation (`src/eval/content_preservation.py`) plus a cheat
  detector for the spec's headline failure mode — passes the tests while
  ignoring the input
- Alternate implementations (`problems/*/solution_alt.py`) for four problems:
  genuinely different algorithms that pass the same tests, so the cheat
  detector is *validated* rather than merely asserted

Next: LoRA fine-tune (Week 2 proper). Blocked on a data-volume decision —
see "Honest status" below.

## What the eval already showed

Two findings, both worth more than the code:

**Functional correctness alone is not a result.** The `identity` baseline —
which returns its input unchanged — scores 100% functional correctness and a
perfect content score. So does `normalize_only`. Any headline number that a
do-nothing baseline ties is not measuring the task.

**Naive content preservation cannot detect cheating.** Scored on raw source,
the legitimate and cheating populations *overlap*: the most aggressive real
rewrite in the training data scored 0.519 while the hardest deliberate cheat
(brute-force max_subarray standing in for Kadane's) scored 0.555. No threshold
separates them.

The fix was not a better threshold but a **style-invariant comparison**
(`src/eval/canonicalize.py`): reduce both sides to a style-normal form first,
so differences the style transforms can undo stop reading as content changes.
Loop idioms are canonicalized in the *easy* direction — recovering a `for`
from an arbitrary `while` is undecidable, so both sides are pushed to the
`while` form instead, which is all invariance requires.

| content metric | legit rewrites (min) | cheats (max) | separation |
|---|---|---|---|
| raw source | 0.519 | 0.555 | **−0.036 (overlap)** |
| style-invariant | 0.963 | 0.477 | **+0.485** |

With that margin the 0.70 threshold sits mid-gap rather than being fitted to
examples, and all four deliberate cheats are caught at 100% with no false
positives on legitimate rewriters.

Reproduce: `python src/eval/run_baselines.py --split train`

## Honest status and known limits

- **The corpus is far too small to train on.** 204 directed examples from 10
  problems, where the spec calls for thousands. Variants of ten problems are
  also heavily correlated, so they would mostly teach the model these ten
  programs. Scaling the corpus is the blocking task before any fine-tune —
  more transforms would multiply examples without adding real diversity.
- **The correctness gate has never rejected anything** (0 of 102 variants
  failed). That is expected — the transforms are AST-safe by construction —
  but it means the gate is currently unexercised. It starts doing real work
  when a *model* generates the rewrites.
- **`to_verbose` is barely implemented.** The rule-based baseline scores 48%
  on style hit, which is essentially "wins the to_terse half, does nothing on
  the to_verbose half." There is no transform that adds comments or expands
  names, so the verbose direction has no rule-based floor to beat.
- **The cheat detector is validated on four hand-written examples.** The
  control-flow term in particular was added *because* max_subarray slipped
  past, so it is fitted to a known failure, not independently confirmed.
- **Comments are not a live style axis.** Every side goes through
  `ast.unparse`, which drops `#` comments, so only docstrings survive.
  Preserving real comments needs a concrete-syntax pipeline (libcst) rather
  than `ast`.

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
