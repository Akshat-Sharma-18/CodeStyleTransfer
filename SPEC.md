# Code Style Transfer

A project spec for training a model that rewrites code from one style to another — while keeping it correct (it still compiles and passes tests).

---

## What the project actually is

Take a working program and rewrite it in a different **style** without changing its behavior. Concrete framings, easiest to hardest:

- **Verbose ↔ terse:** commented, descriptive-variable, multi-function code ↔ tight competitive-programming style. You already live on both sides of this, which is a real advantage.
- **Formatting/idiom normalization:** rewrite arbitrary code into one consistent house style (naming, loop idioms, early-returns, etc.).
- **Fingerprint transfer (the flashy one):** rewrite code so it reads like a *specific* author wrote it. Fun demo, hardest to evaluate, and has an ethics footnote (see below).

The thing that makes this project good: the output is **checkable**. Behavior preservation isn't a matter of taste — you run the tests. That gives you a hard metric most "style" projects lack.

---

## The central challenge: parallel data

Style transfer usually needs pairs (same program, two styles). You have three ways to get them, and your choice defines the project's difficulty.

### Option A — Mechanical transforms (recommended start)

Generate pairs *automatically* by applying reversible, behavior-preserving transformations:

- rename identifiers (descriptive ↔ `a`, `b`, `x`)
- add/strip comments and whitespace
- expand/inline helper functions
- `for` ↔ `while`, ternary ↔ if/else, list-comprehension ↔ loop
- reorder independent statements

You control both sides, pairs are free, and correctness is guaranteed by construction. The model learns to *undo/redo* these. Weakness: it only learns the transforms you scripted — so be honest in the writeup that it's a bounded style space.

### Option B — Your own submissions

If you have the same problem solved twice (a clean version and a contest version), those are gold real pairs. Realistically you won't have many, so use this as an **eval set**, not training data.

### Option C — Unpaired + round-trip (the ambitious version)

No pairs. Two corpora (verbose code, terse code) and train unsupervised, CycleGAN-style / back-translation-style, with a **round-trip consistency loss**: `terse → verbose → terse` should return the original, and every intermediate must still pass tests. This is the research-grade framing and the strongest resume story — but it's also where you'll spend weeks. Do A first, reach for C only if A is working.

---

## Hardware notes

Target machine: 24GB RAM, RTX 5050 (~8GB), AMD Ryzen AI 7 (8-core).

- 8GB VRAM comfortably fine-tunes a **small code model** (e.g. a ~350M–1B code model) with LoRA/QLoRA. Full fine-tunes of larger models won't fit — use parameter-efficient tuning.
- As with the adversarial project, a big CPU cost is **running the tests** to verify correctness. Parallelize across your 8 cores.
- A seq2seq or decoder-only code model both work; starting from a pretrained code model beats training from scratch here — the model already knows syntax, you're only teaching style.

---

## Architecture

**Backbone:** fine-tune a pretrained small code model (CodeT5/CodeT5+ as an encoder-decoder, or a small code-focused decoder-only LM). LoRA so it fits in 8GB.

**Task format:** prepend a style token/instruction (`<to_terse>` / `<to_verbose>`) to the input, output the rewritten code. One model, both directions.

**The correctness gate (your differentiator):** every generated rewrite is piped through the runner — compile + run the problem's test suite. This is used two ways:

1. as your **eval metric** (% of rewrites that still pass), and
2. optionally as a **reject-sampling / RL reward** to fine-tune toward correctness after supervised training.

---

## Evaluation — this is where the project earns its keep

Report all three; they measure different things:

- **Functional correctness:** % of rewrites that still compile and pass the original tests. The headline number.
- **Style-target hit:** did it actually move toward the target style? Measure with a cheap style classifier you train (verbose vs terse), or concrete proxies (identifier length, comment ratio, line count, token count).
- **Content preservation:** it shouldn't just rewrite into *generic* correct code — it should preserve the original's logic. CodeBLEU or AST edit-distance to the reference (when you have Option B pairs).

The interesting failure mode to write up: models that cheat by producing correct-but-unrelated code (passes tests, ignores the input). Catching and reporting this shows maturity.

---

## Reward-hacking / cheating traps

- **"Passes tests" ≠ "faithful rewrite."** A model can learn to emit a canonical solution it memorized. Guard with content-preservation metrics, not just the test gate.
- **Trivial style wins.** If "terse" is scored only by line count, the model learns to jam everything onto one line illegibly. Use multiple style proxies, or a learned style classifier.
- **Mechanical-transform overfit (Option A).** The model may only handle the exact transforms you scripted. State the bounded style space honestly and test on held-out real pairs (Option B) to expose the gap.

---

## Data plan

- **Train:** Option A mechanical pairs, generated from a corpus of working solutions (your own + open competitive-programming solution sets). Thousands of pairs is plenty for LoRA.
- **Eval (held-out, real):** Option B — your own dual-style solutions. Small but honest; this is the number that actually means something.
- **Every problem** in the corpus needs a **test suite** so the correctness gate works. If you reuse the adversarial project's problem library and runner, you get this for free — the two projects share infrastructure.

---

## "Done" at each stage

- **v1 done:** supervised model on Option A pairs; high functional-correctness on held-out problems in the *same* transform space. Deliverable: correctness + style-shift numbers, before/after code samples.
- **v2 done:** it generalizes to real pairs (Option B eval) better than a naive baseline (e.g. a formatter/linter, or a prompt-only LLM with no fine-tuning). Beating the prompt-only baseline is the key result — it justifies training at all.
- **Stretch:** unpaired round-trip (Option C) with test-passing intermediates, or the fingerprint-transfer demo.

---

## Realistic timeline (part-time)

| Week | Goal |
|------|------|
| Week 1 | Corpus + test suites + the correctness runner. Mechanical transform generator (Option A) producing pairs. |
| Week 2 | LoRA fine-tune of the code backbone; both-direction rewriting; functional-correctness eval working. |
| Week 3 | Style proxies + style classifier; the full three-metric eval table; baseline comparison (prompt-only LLM, formatter). |
| Week 4 | Option B real-pair eval, error analysis (the cheating failure modes), writeup + demo. |
| Week 5+ | Stretch: reject-sampling/RL toward correctness, or Option C round-trip. |

**Honest risk:** the correctness runner and the parallel-data pipeline are more work than the fine-tuning itself. And the "passes tests but ignored the input" cheat *will* show up — budget analysis time for it.

---

## Ethics footnote (for the fingerprint variant)

Rewriting code to imitate a *specific named person's* style shades into deanonymization / impersonation territory (e.g. defeating stylometry that protects anonymous contributors). Keep the fingerprint demo to consenting subjects (you, a willing friend) and say so in the writeup. Framing it as "author-style transfer with consent" instead of "impersonate anyone" is both more responsible and a better interview answer.

---

## Shared infrastructure with the adversarial project

If you build both, the **problem library + test suites + sandboxed runner** are shared. Build that foundation once and both projects sit on top of it — worth sequencing them back-to-back if you want two resume items for not much more than 1.5x the work.
