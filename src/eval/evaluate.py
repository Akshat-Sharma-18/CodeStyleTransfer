"""The three-metric eval table, plus the cheat detector.

Takes any rewriter -- a fine-tuned model, a prompt-only LLM, a formatter, or a
deliberate cheater -- as a plain callable `(code, direction) -> code`, so every
baseline is scored by exactly the same harness.

Metrics (all three reported; they measure different things):
  1. functional correctness -- % of rewrites still passing the original tests
  2. style-target hit       -- % that actually moved toward the requested style
  3. content preservation   -- did it keep the *input's* logic?

And the number that matters most for honesty:
  suspected_cheat_rate -- passes the tests but content preservation is low,
  i.e. the model emitted correct-but-unrelated code. A model can score 100%
  on metric 1 while being useless; this is what catches that.
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from corpus import tests_for  # noqa: E402
from eval.content_preservation import content_score  # noqa: E402
from eval.style_proxies import moved_toward_terse  # noqa: E402
from runner.sandbox import run_against_tests  # noqa: E402

Rewriter = Callable[[str, str], str]

# Below this content score, a passing rewrite is treated as suspicious: it
# satisfied the tests without keeping the input's logic.
#
# Calibration: with style-invariant canonicalization, the 176 verified
# legitimate rewrites score >= 0.963 and the four deliberate cheats score
# <= 0.477. Any threshold in that gap works; 0.70 sits in the middle of it.
# Without canonicalization the two populations overlap (legit min 0.519 vs
# cheat max 0.555) and no threshold separates them at all -- see
# eval/canonicalize.py for why that matters.
CHEAT_CONTENT_THRESHOLD = 0.70


@dataclass
class EvalResult:
    name: str
    total: int = 0
    parsed: int = 0
    passed_tests: int = 0
    style_hits: int = 0
    content_scores: list[float] = field(default_factory=list)
    suspected_cheats: list[dict] = field(default_factory=list)
    # Per-direction, because the two are not the same task: going terse is
    # deletion, going verbose requires inventing information that is not in
    # the input. A single averaged style number hides that completely.
    per_direction: dict[str, dict[str, int]] = field(default_factory=dict)

    def note_direction(self, direction: str, style_hit: bool, passed: bool) -> None:
        bucket = self.per_direction.setdefault(direction, {"n": 0, "style_hits": 0, "passed": 0})
        bucket["n"] += 1
        bucket["style_hits"] += int(style_hit)
        bucket["passed"] += int(passed)

    def direction_style_rate(self, direction: str) -> float | None:
        bucket = self.per_direction.get(direction)
        if not bucket or not bucket["n"]:
            return None
        return bucket["style_hits"] / bucket["n"]

    @property
    def functional_correctness(self) -> float:
        return self.passed_tests / self.total if self.total else 0.0

    @property
    def style_hit_rate(self) -> float:
        return self.style_hits / self.total if self.total else 0.0

    @property
    def mean_content_score(self) -> float:
        return sum(self.content_scores) / len(self.content_scores) if self.content_scores else 0.0

    @property
    def suspected_cheat_rate(self) -> float:
        return len(self.suspected_cheats) / self.total if self.total else 0.0

    def as_row(self) -> dict:
        return {
            "rewriter": self.name,
            "n": self.total,
            "parse_rate": self.parsed / self.total if self.total else 0.0,
            "functional_correctness": self.functional_correctness,
            "style_hit_rate": self.style_hit_rate,
            "mean_content_score": self.mean_content_score,
            "suspected_cheat_rate": self.suspected_cheat_rate,
            "style_hit_to_terse": self.direction_style_rate("to_terse"),
            "style_hit_to_verbose": self.direction_style_rate("to_verbose"),
        }


def load_examples(split_path: Path) -> list[dict]:
    return [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _strip_style_token(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("<to_"):
        return "\n".join(lines[1:])
    return text


def _score_one(rewriter: Rewriter, example: dict) -> dict:
    source = _strip_style_token(example["input"])
    direction = example["direction"]

    try:
        rewrite = rewriter(source, direction)
    except Exception as exc:  # noqa: BLE001 - a rewriter that blew up scores as a failure
        return {"parsed": False, "passed": False, "style_hit": False, "content": None, "error": str(exc)}

    try:
        ast.parse(rewrite)
    except SyntaxError as exc:
        return {"parsed": False, "passed": False, "style_hit": False, "content": None, "error": str(exc)}

    passed = run_against_tests(rewrite, tests_for(example["problem"])).passed

    try:
        if direction == "to_terse":
            style_hit = moved_toward_terse(source, rewrite)
        else:
            style_hit = moved_toward_terse(rewrite, source)
        content = content_score(source, rewrite)
    except SyntaxError as exc:
        return {"parsed": False, "passed": False, "style_hit": False, "content": None, "error": str(exc)}

    return {
        "parsed": True,
        "passed": passed,
        "style_hit": style_hit,
        "content": content,
        "rewrite": rewrite,
        "source": source,
        "problem": example["problem"],
        "direction": direction,
    }


def evaluate(name: str, rewriter: Rewriter, examples: list[dict], workers: int = 8) -> EvalResult:
    result = EvalResult(name=name)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(lambda ex: _score_one(rewriter, ex), examples))

    for outcome in outcomes:
        result.total += 1
        if not outcome["parsed"]:
            continue

        result.parsed += 1
        if outcome["passed"]:
            result.passed_tests += 1
        if outcome["style_hit"]:
            result.style_hits += 1
        result.note_direction(outcome["direction"], outcome["style_hit"], outcome["passed"])

        content = outcome["content"]
        result.content_scores.append(content["content_score"])

        # The cheat signature: tests pass, but the output doesn't resemble the input.
        if outcome["passed"] and content["content_score"] < CHEAT_CONTENT_THRESHOLD:
            result.suspected_cheats.append(
                {
                    "problem": outcome["problem"],
                    "direction": outcome["direction"],
                    "content_score": round(content["content_score"], 3),
                    "rewrite": outcome["rewrite"],
                }
            )

    return result


def format_table(results: list[EvalResult]) -> str:
    headers = [
        ("rewriter", 24),
        ("n", 5),
        ("parse", 7),
        ("correct", 9),
        ("style", 7),
        ("content", 9),
        ("cheat?", 7),
        ("->terse", 8),
        ("->verbose", 9),
    ]
    lines = ["  ".join(name.ljust(width) for name, width in headers)]
    lines.append("-" * (sum(width for _, width in headers) + 2 * (len(headers) - 1)))

    def rate(value: float | None, width: int) -> str:
        return ("--" if value is None else f"{value:.0%}").ljust(width)

    for result in results:
        row = result.as_row()
        lines.append(
            "  ".join(
                [
                    str(row["rewriter"]).ljust(24),
                    str(row["n"]).ljust(5),
                    f"{row['parse_rate']:.0%}".ljust(7),
                    f"{row['functional_correctness']:.0%}".ljust(9),
                    f"{row['style_hit_rate']:.0%}".ljust(7),
                    f"{row['mean_content_score']:.3f}".ljust(9),
                    f"{row['suspected_cheat_rate']:.0%}".ljust(7),
                    rate(result.direction_style_rate("to_terse"), 8),
                    rate(result.direction_style_rate("to_verbose"), 9),
                ]
            )
        )
    return "\n".join(lines)
