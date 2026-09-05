"""Ingest HumanEval into the corpus.

MBPP gave the corpus scale; HumanEval is here for *style*, which is the gap
that actually limits this project. MBPP solutions arrive with essentially no
docstrings and no type annotations, so two of the seven transforms
(`strip_comments`, `strip_annotations`) never fire on them -- which is why the
terse direction had so little to delete and the rule-based baseline hit its
style target only 15% of the time going terse.

HumanEval problems are the opposite: each `prompt` is a signature with type
hints plus a worked docstring, and the full solution is `prompt +
canonical_solution`. That is precisely the verbose end of the axis the corpus
was missing. It is small (164 problems), but every one of them exercises the
transforms MBPP cannot.

Its tests are a `check(candidate)` function rather than bare assertions, so the
generated module calls `check` against the problem's entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.ingest_common import verify_and_write  # noqa: E402


def build_test_module(test_code: str, entry_point: str) -> str:
    return f"from solution import *\n\n{test_code}\n\ndef test_check():\n    check({entry_point})\n"


def prepare(record: dict) -> dict | None:
    prompt = record.get("prompt") or ""
    body = record.get("canonical_solution") or ""
    entry_point = record.get("entry_point")
    test_code = record.get("test") or ""
    if not prompt or not body or not entry_point or not test_code:
        return None

    return {
        "name": f"humaneval_{record['task_id'].split('/')[-1]}",
        "solution": prompt + body,
        "tests": build_test_module(test_code, entry_point),
        "statement": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset("openai/openai_humaneval")
    records = [r for split in dataset for r in dataset[split]]

    problems = [p for p in (prepare(r) for r in records) if p is not None]
    verify_and_write(problems, "humaneval.jsonl", workers=args.workers, timeout=args.timeout)


if __name__ == "__main__":
    main()
