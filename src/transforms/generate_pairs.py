"""Generate Option A (mechanical) training pairs from the problem corpus.

For each problem, apply each transform to the original solution, verify the
transformed code still passes the problem's tests via the correctness gate,
and write (original, transformed, transform_name) pairs to data/pairs/.

Only pairs that pass verification are kept -- correctness by construction is
the whole point of Option A.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from runner.sandbox import run_against_tests  # noqa: E402
from transforms.comments import strip_comments_and_docstrings  # noqa: E402
from transforms.loops import for_range_to_while  # noqa: E402
from transforms.rename import terse_rename  # noqa: E402
from transforms.ternary import if_else_to_ternary  # noqa: E402

TRANSFORMS = {
    "strip_comments": strip_comments_and_docstrings,
    "for_range_to_while": for_range_to_while,
    "terse_rename": lambda code: terse_rename(code).code,
    "if_else_to_ternary": if_else_to_ternary,
}


def generate() -> list[dict]:
    problems_dir = ROOT / "problems"
    pairs_dir = ROOT / "data" / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for problem_dir in sorted(problems_dir.iterdir()):
        solution_path = problem_dir / "solution.py"
        test_path = problem_dir / "test_solution.py"
        if not solution_path.exists() or not test_path.exists():
            continue

        original_code = solution_path.read_text(encoding="utf-8")
        test_code = test_path.read_text(encoding="utf-8")
        normalized_original = ast.unparse(ast.parse(original_code))

        for transform_name, transform_fn in TRANSFORMS.items():
            try:
                transformed_code = transform_fn(original_code)
            except Exception as exc:  # noqa: BLE001 - log and skip malformed transforms
                print(f"[skip] {problem_dir.name}/{transform_name}: {exc}")
                continue

            if transformed_code == normalized_original:
                print(f"[skip] {problem_dir.name}/{transform_name}: no-op (pattern didn't match)")
                continue

            result = run_against_tests(transformed_code, test_code)
            status = "kept" if result.passed else "dropped (failed tests)"
            print(f"[{status}] {problem_dir.name}/{transform_name}")

            if result.passed:
                all_pairs.append(
                    {
                        "problem": problem_dir.name,
                        "transform": transform_name,
                        "verbose": original_code,
                        "terse": transformed_code,
                    }
                )

    out_path = pairs_dir / "mechanical_pairs.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nWrote {len(all_pairs)} verified pairs to {out_path}")
    return all_pairs


if __name__ == "__main__":
    generate()
