"""Score every baseline on the validation split and print the eval table.

Also runs the cheat-detector validation: for each problem that has an
alternate implementation, score a rewriter that emits that alternate and
confirm the harness flags it despite 100% test passing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval.baselines import BASELINES, make_canonical_cheat  # noqa: E402
from eval.evaluate import evaluate, format_table, load_examples  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    split_path = ROOT / "data" / "pairs" / f"{args.split}.jsonl"
    examples = load_examples(split_path)
    print(f"Scoring {len(examples)} examples from {split_path.name}\n")

    results = [evaluate(name, fn, examples, args.workers) for name, fn in BASELINES.items()]

    # Cheat validation: only over problems that actually have an alternate.
    alt_problems = sorted(p.parent.name for p in (ROOT / "problems").glob("*/solution_alt.py"))
    cheat_examples = [ex for ex in examples if ex["problem"] in alt_problems]
    if cheat_examples:
        by_problem: dict[str, list[dict]] = {}
        for example in cheat_examples:
            by_problem.setdefault(example["problem"], []).append(example)
        for problem, problem_examples in by_problem.items():
            results.append(
                evaluate(f"cheat[{problem}]", make_canonical_cheat(problem), problem_examples, args.workers)
            )

    print(format_table(results))

    print("\nWhat to read here:")
    print("  identity scores 100% correctness while doing nothing -- correctness alone is not a result.")
    if cheat_examples:
        print("  cheat[*] rows pass the tests too; the content score and cheat rate are what separate them.")
    else:
        print("  (no val problem has an alternate solution -- run with --split train to exercise the cheat check)")

    out_path = ROOT / "data" / "eval" / f"baselines_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([r.as_row() for r in results], indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
