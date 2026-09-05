"""Shared ingestion machinery for external problem sources.

Every source has to clear the same bar before it enters the corpus: the
problem must parse, and its *own reference solution* must pass its *own tests*.
A problem that fails that check is worse than useless here -- it would
manufacture false failures for every model later evaluated against it. MBPP
had 7 such problems.
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from runner.sandbox import run_against_tests  # noqa: E402

CORPUS_DIR = ROOT / "data" / "corpus"


def parses(source: str) -> bool:
    try:
        ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    return True


def verify_and_write(
    problems: list[dict],
    out_name: str,
    workers: int = 8,
    timeout: float = 15.0,
    progress: Callable[[str], None] = print,
) -> dict:
    """Drop problems whose reference solution fails its own tests, then write."""
    usable = [p for p in problems if parses(p["solution"]) and parses(p["tests"])]
    progress(f"{len(problems)} records -> {len(usable)} parseable with tests")

    progress(f"verifying reference solutions against their own tests ({workers} workers)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = list(pool.map(lambda p: run_against_tests(p["solution"], p["tests"], timeout).passed, usable))

    verified = [p for p, ok in zip(usable, verdicts) if ok]
    rejected = len(usable) - len(verified)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CORPUS_DIR / out_name
    with out_path.open("w", encoding="utf-8") as f:
        for problem in verified:
            f.write(json.dumps(problem) + "\n")

    progress(f"\nkept {len(verified)}, dropped {rejected} whose reference solution failed its own tests")
    progress(f"wrote {out_path}")
    return {"records": len(problems), "usable": len(usable), "verified": len(verified), "rejected": rejected}
