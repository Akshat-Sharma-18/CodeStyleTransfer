"""Ingest MBPP into the corpus.

MBPP (Mostly Basic Python Problems, google-research-datasets/mbpp) is a good
fit for this project specifically because every problem ships executable
assertions -- the correctness gate needs a test suite per problem, and writing
~900 of them by hand is not happening.

Two format mismatches have to be bridged:

  * MBPP tests call bare function names (`assert max_chain_length(...) == 3`),
    not a fixed `solve` entry point, and some depend on classes defined in the
    solution. So the generated test module does `from solution import *`.
  * Some problems need `test_setup_code` executed first.

Every ingested problem is then run through the correctness gate with its own
reference solution, and dropped if that fails. A problem whose *reference*
answer doesn't pass its own tests is worse than useless here: it would
manufacture false failures for every model we later evaluate.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from runner.sandbox import run_against_tests  # noqa: E402

CORPUS_DIR = ROOT / "data" / "corpus"


def build_test_module(test_list: list[str], setup_code: str) -> str:
    """Turn MBPP's bare assertions into an importable pytest module."""
    lines = ["from solution import *", ""]
    if setup_code and setup_code.strip():
        lines.extend([setup_code.strip(), ""])

    for index, assertion in enumerate(test_list):
        lines.append(f"def test_{index}():")
        for assertion_line in assertion.strip().splitlines():
            lines.append(f"    {assertion_line}")
        lines.append("")

    return "\n".join(lines)


def _prepare(record: dict) -> dict | None:
    solution = record["code"]
    tests = record.get("test_list") or []
    if not tests:
        return None

    # Must be parseable -- every transform in this project is AST-based.
    try:
        ast.parse(solution)
    except SyntaxError:
        return None

    test_code = build_test_module(tests, record.get("test_setup_code", ""))
    try:
        ast.parse(test_code)
    except SyntaxError:
        return None

    return {
        "name": f"mbpp_{record['task_id']}",
        "solution": solution,
        "tests": test_code,
        "statement": record.get("text", ""),
    }


def _verify(problem: dict, timeout: float) -> bool:
    return run_against_tests(problem["solution"], problem["tests"], timeout_s=timeout).passed


def ingest(limit: int | None, workers: int, timeout: float) -> dict:
    from datasets import load_dataset

    dataset = load_dataset("google-research-datasets/mbpp", "full")

    records = []
    for split in dataset:
        records.extend(list(dataset[split]))
    if limit:
        records = records[:limit]

    prepared = [p for p in (_prepare(r) for r in records) if p is not None]
    print(f"{len(records)} MBPP records -> {len(prepared)} parseable with tests")

    print(f"verifying reference solutions against their own tests ({workers} workers)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(lambda p: _verify(p, timeout), prepared))

    verified = [p for p, ok in zip(prepared, verdicts) if ok]
    rejected = len(prepared) - len(verified)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPUS_DIR / "mbpp.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for problem in verified:
            f.write(json.dumps(problem) + "\n")

    print(f"\nkept {len(verified)}, dropped {rejected} whose reference solution failed its own tests")
    print(f"wrote {out_path}")
    return {"records": len(records), "prepared": len(prepared), "verified": len(verified), "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only ingest the first N records (for smoke tests)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    ingest(args.limit, args.workers, args.timeout)


if __name__ == "__main__":
    main()
