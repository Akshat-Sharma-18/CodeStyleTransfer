"""Unified view over the problem corpus.

Two sources, one interface:

  problems/<name>/          hand-written problems, one directory each. Small,
                            readable, and the place to add cases by hand.
  data/corpus/*.jsonl       ingested problems (e.g. MBPP). Thousands of tiny
                            directories would be unpleasant in git, so bulk
                            sources live in a single file per source.

Everything downstream -- pair generation, the correctness gate, the eval
harness -- goes through `load_corpus()` and never touches either layout
directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS_DIR = ROOT / "problems"
CORPUS_DIR = ROOT / "data" / "corpus"


@dataclass(frozen=True)
class Problem:
    name: str
    solution: str
    tests: str
    source: str

    @property
    def alt_solution_path(self) -> Path:
        return PROBLEMS_DIR / self.name / "solution_alt.py"

    @property
    def has_alternate(self) -> bool:
        return self.alt_solution_path.exists()


def _load_handwritten() -> list[Problem]:
    problems = []
    if not PROBLEMS_DIR.exists():
        return problems

    for problem_dir in sorted(PROBLEMS_DIR.iterdir()):
        solution_path = problem_dir / "solution.py"
        test_path = problem_dir / "test_solution.py"
        if not solution_path.exists() or not test_path.exists():
            continue
        problems.append(
            Problem(
                name=problem_dir.name,
                solution=solution_path.read_text(encoding="utf-8"),
                tests=test_path.read_text(encoding="utf-8"),
                source="handwritten",
            )
        )
    return problems


def _load_ingested() -> list[Problem]:
    problems = []
    if not CORPUS_DIR.exists():
        return problems

    for corpus_file in sorted(CORPUS_DIR.glob("*.jsonl")):
        source = corpus_file.stem
        for line in corpus_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            problems.append(
                Problem(
                    name=record["name"],
                    solution=record["solution"],
                    tests=record["tests"],
                    source=source,
                )
            )
    return problems


@lru_cache(maxsize=1)
def load_corpus() -> tuple[Problem, ...]:
    """All problems from every source, hand-written first."""
    return tuple(_load_handwritten() + _load_ingested())


@lru_cache(maxsize=1)
def _by_name() -> dict[str, Problem]:
    return {problem.name: problem for problem in load_corpus()}


def get_problem(name: str) -> Problem:
    return _by_name()[name]


def tests_for(name: str) -> str:
    return get_problem(name).tests
