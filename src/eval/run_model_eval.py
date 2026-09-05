"""Score the fine-tuned model against the baselines it has to beat.

This produces the project's v2 result. The comparison that matters is not
"the model scores X" but the three rows together:

  rule_based   what mechanical transforms alone achieve -- no learning
  base         the backbone, prompted but never fine-tuned (the prompt-only
               baseline the spec says is the bar worth clearing)
  adapter      the same backbone with the LoRA adapter

If `adapter` does not beat `base`, the fine-tune bought nothing and the honest
conclusion is to say so. If it beats `base` on style but its content score
falls or its cheat rate climbs, it learned to emit plausible-looking code
rather than to rewrite the input -- which the harness is built to surface
rather than hide.

Model rows are slow: every example is a beam-search generation followed by a
subprocess test run, so `--limit` is there for iterating.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.baselines import BASELINES  # noqa: E402
from eval.evaluate import evaluate, format_table, load_examples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--adapter", default=str(ROOT / "data" / "checkpoints" / "lora"))
    parser.add_argument("--model", default="Salesforce/codet5p-220m")
    parser.add_argument("--limit", type=int, default=200, help="examples to score (model rows are slow)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    examples = load_examples(ROOT / "data" / "pairs" / f"{args.split}.jsonl")
    if args.limit and args.limit < len(examples):
        # Sample rather than truncate: the file is grouped by problem, so the
        # first N examples would cover only a handful of problems.
        examples = random.Random(args.seed).sample(examples, args.limit)
    print(f"scoring {len(examples)} examples from {args.split}.jsonl\n")

    results = [evaluate("rule_based", BASELINES["rule_based"], examples, args.workers)]

    from model.rewriter import ModelRewriter

    if not args.skip_base:
        print("loading base model (no adapter)...")
        base = ModelRewriter(model_name=args.model, num_beams=args.num_beams)
        # Generation is GPU-serialized; parallel workers only add contention.
        results.append(evaluate("base (prompt-only)", base, examples, workers=1))

    adapter_path = Path(args.adapter)
    if adapter_path.exists():
        print("loading fine-tuned adapter...")
        tuned = ModelRewriter(model_name=args.model, adapter_path=str(adapter_path), num_beams=args.num_beams)
        results.append(evaluate("adapter (fine-tuned)", tuned, examples, workers=1))
    else:
        print(f"no adapter at {adapter_path} -- train one with src/model/train_lora.py")

    print()
    print(format_table(results))

    out_path = ROOT / "data" / "eval" / f"model_eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "n": len(examples),
        "rows": [r.as_row() for r in results],
        "suspected_cheats": {r.name: r.suspected_cheats[:5] for r in results if r.suspected_cheats},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")

    for result in results:
        if result.suspected_cheats:
            print(f"\n{result.name}: {len(result.suspected_cheats)} suspected cheats, lowest-content example:")
            worst = min(result.suspected_cheats, key=lambda c: c["content_score"])
            print(f"  {worst['problem']} ({worst['direction']}, content {worst['content_score']}):")
            for line in worst["rewrite"].splitlines()[:12]:
                print(f"    {line}")


if __name__ == "__main__":
    main()
