# Code Style Transfer

Fine-tune a small code model to rewrite code between styles (verbose ↔ terse, idiom
normalization, and eventually author-fingerprint transfer) while preserving behavior.
Every rewrite is checked by actually running the original test suite against it —
correctness is a hard metric here, not a matter of taste.

See [SPEC.md](SPEC.md) for the full project spec.

## Status

Week 1 complete:

- **1,141-problem corpus**: 10 hand-written, 967 from MBPP, 164 from
  HumanEval — every one carrying an executable test suite. `src/corpus.py`
  presents all three behind one interface. The two sources are stylistically
  complementary and deliberately so: MBPP supplies terse, undocumented code
  (0.0% docstrings), HumanEval supplies the documented end (99.4% docstrings,
  36.6% annotations). HumanEval yields 10.4 verified style variants per
  problem against MBPP's 3.2, precisely because the terse-direction transforms
  finally have something to remove
- **Correctness runner** (`src/runner/sandbox.py`): runs candidate code
  against a problem's pytest suite in an isolated subprocess
- **Seven mechanical transforms**, all AST-based: identifier renaming,
  docstring stripping, type-annotation stripping, `for range(...)` → `while`,
  `if/else` → ternary, augmented-assignment expansion, comprehension expansion
- **6,216 verified pairs** (4,936 train / 1,280 val) in `data/pairs/`, split
  by problem so validation problems are entirely unseen
- 35 self-tests (`tests/`)

Week 2 complete: LoRA fine-tune of CodeT5+ 220M (`src/model/train_lora.py`),
with the model wrapped as the same `(code, direction) -> code` callable the
baselines use (`src/model/rewriter.py`) so it is scored by an identical
harness. See **Result** below.

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

## Result

120 sampled validation examples, beam search (4 beams), every row scored by the
same harness:

| rewriter | parse | correct | style | content | cheat? | →terse | →verbose |
|---|---|---|---|---|---|---|---|
| rule_based | 100% | **100%** | 34% | 0.999 | 0% | 46% | 22% |
| base (prompt-only) | 17% | 0% | 10% | 0.256 | 0% | 46%* | 86%* |
| adapter (fine-tuned) | 99% | **95%** | 31% | 0.984 | 0% | 38% | 24% |

<sub>*The base row's per-direction figures were computed over its parsed 17%
only, so they overstate badly — 86% of almost nothing. That inconsistency is
fixed in the harness (per-direction now counts every attempt, unparsed as a
miss); the table is left as the run produced it rather than silently restating
numbers that were not measured. The other two rows parse at 99–100%, where the
distinction is immaterial.</sub>

**The fine-tune clears the spec's v2 bar.** Against the prompt-only backbone it
goes 0% → 95% functional correctness, 17% → 99% parse rate, and 0.256 → 0.984
content preservation. Prompting a 220M code model to do this task does not
work; training it does. That comparison is what justifies training at all.

**It does not beat the rule-based baseline.** The honest reading is that it
roughly matches it while losing slightly on every axis: 95% vs 100% correct,
31% vs 34% style, 0.984 vs 0.999 content. A model that merely ties a scripted
transform is not yet a reason to prefer the model. Its case has to be
generalization beyond the seven scripted transforms — which is precisely what
the Option B held-out real pairs are for, and what this corpus cannot yet show.

Cheat rate is 0% on every row and content scores sit far above the 0.70
threshold, so the model is genuinely rewriting its input rather than emitting
memorized solutions.

Reproduce: `python src/eval/run_model_eval.py --split val --limit 120`

### The training bug this table found

The first fine-tune scored 45% parse, 27% correct, 0.595 content, and tripped
the cheat detector on 4% of examples — its worst output was the same function
emitted six times with renamed variables. The cause was not the model but the
tokenizer workaround: building the BPE backend from raw `vocab.json` +
`merges.txt` drops the post-processing template, so `tokenizer(target)`
returned no `</s>`, seq2seq labels carried no EOS, and the model was never
taught where to stop.

Nothing crashed, no test failed, and **training loss fell the whole way**
(0.82 train / 0.15 eval). The bug was visible only as a mediocre score against
a non-learned baseline. That is the argument for building the eval harness
before the model rather than reading loss curves.

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
positives on legitimate rewriters. The separation still holds after the corpus
grew a hundredfold — across all 3,278 real MBPP training pairs the lowest
legitimate score is 0.843 against a cheat maximum of 0.457 — so it is a
property of the metric, not of the ten problems it was designed on. It is
asserted as a test, and fails loudly if a change collapses it.

Reproduce: `python src/eval/run_baselines.py --split train`

**The transforms were not as behavior-preserving as "AST-based" suggests.**
Running them over 977 real problems, the correctness gate rejected 39 rewrites
that were silently wrong. Every one was a genuine bug, and none would have been
obvious by reading the code:

| transform | bug | resolution |
|---|---|---|
| `for_range_to_while` | `range(n, -1, -1)` counts down, but the emitted condition was always `i < stop`, so countdown loops became no-ops | pick `<`/`>` from the step's sign; refuse non-literal steps |
| `for_range_to_while` | the increment sits at the end of the body, so `continue` skipped it — an infinite loop | refuse loops with a `continue` bound to them |
| `for_range_to_while` | `range(len(xs))` fixes its bound once; `while i < len(xs)` re-reads it every iteration | hoist the bound into a fresh variable |
| `for_range_to_while` | `for` leaves the index at its last value, `while` one past the end — mbpp_819 reads `lists[i+1]` after the loop | refuse when the index is read outside the body |
| `expand_comprehension` | `xs = [t for t in xs]` became `xs = []` then iterated the now-empty `xs` | refuse self-referencing comprehensions |
| `expand_comprehension` | comprehensions have their own scope, `for` loops do not, so expansion leaked the loop variable | refuse when the name is bound elsewhere |
| `expand_aug_assign` | `list += tuple` works via `__iadd__`; `list = list + tuple` is a `TypeError` | fire only where the target has local evidence of being numeric |

Failures fell from **39 to 6 out of 2,019** (0.3%), and each fix is pinned by a
regression test in `tests/test_transform_safety.py`. This is the argument for
the correctness gate in miniature: seven real semantic bugs, none found by
inspection, all found by running the tests.

**The correctness gate itself was flaky, which is worse than it sounds.** A
5-second timeout meant a legitimately slow solution (mbpp_123 takes 3.4s alone)
failed under 12-way parallel verification purely from CPU contention. Training
pairs were being dropped based on machine load, and eval numbers were not
reproducible run to run. Timeouts are now retried once, serially, with a longer
deadline, and reported separately from real test failures.

## Honest status and known limits

- **The style space is bounded by the seven scripted transforms.** That is
  inherent to the spec's Option A, and the writeup should say so rather than
  implying the model learned "terseness" in general. The corpus now spans both
  ends of the axis, but only along the axes the transforms encode.
- **The style numbers are partly a statement about the corpus.** On MBPP alone
  the rule-based baseline scored 15% going terse against 36% going verbose —
  backwards from the obvious story. The diagnosis was that 0.0% of MBPP
  solutions carry a docstring or annotation, so the terse transforms had
  nothing to delete. Adding HumanEval (99.4% docstrings) tested that
  prediction: terse went **15% → 45%** and the ordering flipped back. A single
  averaged style figure would have hidden the entire effect.
- **The cheat detector's threshold is calibrated, not derived.** The
  separation holds across 3,278 real pairs, but the four cheats it is scored
  against are still hand-written, and the control-flow term was added
  *because* max_subarray slipped past — fitted to a known failure.
- **Comments are not a live style axis.** Every side goes through
  `ast.unparse`, which drops `#` comments, so only docstrings survive.
  Preserving real comments needs a concrete-syntax pipeline (libcst) rather
  than `ast`.
- **The CodeT5 tokenizer needs a workaround to load at all.** Every CodeT5
  checkpoint still ships the legacy `{"content": ..., "__type": "AddedToken"}`
  config form, which transformers 5.x refuses (`TypeError: Input must be a
  List[Union[str, AddedToken]]`) on both the fast and slow paths, with no
  `tokenizer.json` to load directly and a fast-conversion fallback that has to
  instantiate the very slow tokenizer that crashes.
  `src/model/tokenizer_compat.py` builds the byte-level BPE backend straight
  from `vocab.json` + `merges.txt` instead, which round-trips Python source
  exactly. The alternative was pinning transformers to 4.x project-wide.
- **The model ties the rule-based baseline rather than beating it.** It
  decisively beats prompt-only, which is the spec's stated bar, but a scripted
  transform still matches it on in-distribution data. The case for the model
  has to be generalization beyond the scripted transform space, and that needs
  the Option B held-out real pairs the corpus does not yet have.
- **The eval samples 120 examples, not the full 1,280-example split.** Beam
  search runs at roughly 20s per example, so the full split would take hours.
  The reported rates therefore carry sampling noise on the order of a few
  percent, which is wide enough that the 95%-vs-100% correctness gap is real
  but the 31%-vs-34% style gap is not clearly outside it.

## Reproducing the data

The corpus and generated pairs are derived from MBPP and are not checked in.
Rebuild them with:

```bash
python src/data/ingest_mbpp.py --workers 8
python src/data/ingest_humaneval.py --workers 8
python src/transforms/generate_pairs.py --max-subset-size 3 --workers 8
python src/eval/run_baselines.py --split val
```

Ingestion takes a few minutes (it runs every reference solution against its
own tests and drops the ones that fail — 7 of 974 did). Pair generation is the
slow step, since it verifies every candidate rewrite through the test runner.

MBPP is *Program Synthesis with Large Language Models*, Austin et al. 2021,
distributed as `google-research-datasets/mbpp` under CC-BY-4.0. HumanEval is
*Evaluating Large Language Models Trained on Code*, Chen et al. 2021,
distributed as `openai/openai_humaneval` under the MIT licence.

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
