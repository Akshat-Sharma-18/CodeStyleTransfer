"""Build the Option A (mechanical) dataset from the problem corpus.

Pipeline:
  1. For every problem, apply every non-empty *subset* of the transform set
     (in a fixed canonical order) to produce style variants.
  2. Drop no-ops and deduplicate variants by source text -- many subsets
     collapse to the same output, and verifying duplicates is wasted work.
  3. Verify every surviving variant through the correctness gate, in parallel
     across cores (test execution is the expensive step, as the spec warns).
  4. Orient each (original, variant) pair using the style proxies rather than
     assuming the variant is the terse side -- some transforms (comprehension
     expansion, augmented-assignment expansion) make code *more* verbose.
  5. Emit two directed training examples per pair, one per style token.
  6. Split by *problem* so validation problems are entirely unseen.

Only variants that still pass the original tests are kept, so correctness of
the training data is true by construction.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corpus import Problem, load_corpus  # noqa: E402
from eval.style_proxies import moved_toward_terse  # noqa: E402
from runner.sandbox import run_against_tests  # noqa: E402
from transforms.annotations import strip_annotations  # noqa: E402
from transforms.augassign import expand_aug_assign  # noqa: E402
from transforms.comments import strip_comments_and_docstrings  # noqa: E402
from transforms.comprehension import expand_comprehension  # noqa: E402
from transforms.loops import for_range_to_while  # noqa: E402
from transforms.rename import terse_rename  # noqa: E402
from transforms.ternary import if_else_to_ternary  # noqa: E402

# Canonical application order. Composition order matters (renaming last keeps
# the earlier structural transforms readable while they run).
TRANSFORMS: dict[str, callable] = {
    "strip_comments": strip_comments_and_docstrings,
    "strip_annotations": strip_annotations,
    "expand_comprehension": expand_comprehension,
    "expand_aug_assign": expand_aug_assign,
    "for_range_to_while": for_range_to_while,
    "if_else_to_ternary": if_else_to_ternary,
    "terse_rename": lambda code: terse_rename(code).code,
}

TO_TERSE = "<to_terse>"
TO_VERBOSE = "<to_verbose>"


def _apply_subset(source: str, names: tuple[str, ...]) -> str | None:
    """Apply the named transforms in canonical order; None if any of them fails."""
    code = source
    for name in TRANSFORMS:
        if name in names:
            try:
                code = TRANSFORMS[name](code)
            except Exception:  # noqa: BLE001 - a transform that can't handle this code
                return None
    return code


def _collect_candidates(problem: Problem, max_subset_size: int) -> tuple[str, dict[str, tuple[str, ...]]]:
    """Return (original_code, {variant_code: transform_names})."""
    # Round-trip the original through ast.unparse too. Every variant is
    # unparsed output, so leaving the original as raw file text would leak a
    # trivial cue (quote style, spacing) that correlates perfectly with the
    # style label -- the model would learn the formatter, not the style.
    original_code = ast.unparse(ast.parse(problem.solution))

    variants: dict[str, tuple[str, ...]] = {}
    names = list(TRANSFORMS)
    for size in range(1, min(max_subset_size, len(names)) + 1):
        for subset in combinations(names, size):
            variant = _apply_subset(original_code, subset)
            if variant is None or variant == original_code:
                continue
            # Keep the shortest recipe that produces a given variant.
            if variant not in variants or len(subset) < len(variants[variant]):
                variants[variant] = subset

    return original_code, variants


def _process_problem(problem: Problem, max_subset_size: int) -> tuple[list[dict], int]:
    """Generate, verify and orient every variant for one problem."""
    try:
        original_code, variants = _collect_candidates(problem, max_subset_size)
    except SyntaxError:
        return [], 0

    kept = []
    failed = 0
    for variant_code, recipe in variants.items():
        if not run_against_tests(variant_code, problem.tests).passed:
            failed += 1
            continue

        # Let the style proxies decide which side is the verbose one.
        try:
            if moved_toward_terse(original_code, variant_code):
                verbose_code, terse_code = original_code, variant_code
            else:
                verbose_code, terse_code = variant_code, original_code
        except SyntaxError:
            continue

        kept.append(
            {
                "problem": problem.name,
                "transforms": list(recipe),
                "verbose": verbose_code,
                "terse": terse_code,
            }
        )
    return kept, failed


def build(max_subset_size: int, val_fraction: float, seed: int, workers: int) -> dict:
    pairs_dir = ROOT / "data" / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    problems = load_corpus()
    stats = {"problems": len(problems), "candidates": 0, "verified": 0, "failed_verification": 0}
    pairs_by_problem: dict[str, list[dict]] = {}

    # Parallelize across problems: the test runs dominate, and one problem's
    # variants are a small batch on their own.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(lambda p: _process_problem(p, max_subset_size), problems))

    for problem, (kept, failed) in zip(problems, outcomes):
        stats["candidates"] += len(kept) + failed
        stats["failed_verification"] += failed
        stats["verified"] += len(kept)
        if kept:
            pairs_by_problem[problem.name] = kept

    # Split by problem so validation problems are entirely unseen.
    problem_names = sorted(pairs_by_problem)
    random.Random(seed).shuffle(problem_names)
    val_count = max(1, round(len(problem_names) * val_fraction))
    val_problems = set(problem_names[:val_count])

    splits: dict[str, list[dict]] = {"train": [], "val": []}
    for name, pairs in pairs_by_problem.items():
        split = "val" if name in val_problems else "train"
        for pair in pairs:
            # One directed example per style token -- one model, both directions.
            splits[split].append(
                {
                    "problem": pair["problem"],
                    "transforms": pair["transforms"],
                    "direction": "to_terse",
                    "input": f"{TO_TERSE}\n{pair['verbose']}",
                    "target": pair["terse"],
                }
            )
            splits[split].append(
                {
                    "problem": pair["problem"],
                    "transforms": pair["transforms"],
                    "direction": "to_verbose",
                    "input": f"{TO_VERBOSE}\n{pair['terse']}",
                    "target": pair["verbose"],
                }
            )

    for split_name, examples in splits.items():
        out_path = pairs_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        print(f"\n{split_name}: {len(examples)} examples -> {out_path}")

    stats["val_problems"] = sorted(val_problems)
    stats["train_examples"] = len(splits["train"])
    stats["val_examples"] = len(splits["val"])
    (pairs_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        f"\n{stats['candidates']} candidates, {stats['verified']} verified "
        f"({stats['failed_verification']} dropped for failing tests)"
    )
    print(f"held-out val problems: {', '.join(stats['val_problems'])}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-subset-size", type=int, default=4, help="max transforms composed at once")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8, help="parallel test-runner workers")
    args = parser.parse_args()
    build(args.max_subset_size, args.val_fraction, args.seed, args.workers)


if __name__ == "__main__":
    main()
