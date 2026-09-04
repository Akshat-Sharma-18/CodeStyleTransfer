import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.canonicalize import canonicalize
from eval.content_preservation import content_score

ITERATIVE_FIB = "def solve(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"
RECURSIVE_FIB = (
    "def solve(n):\n"
    "    memo = {0: 0, 1: 1}\n"
    "    def helper(k):\n"
    "        if k in memo:\n"
    "            return memo[k]\n"
    "        memo[k] = helper(k - 1) + helper(k - 2)\n"
    "        return memo[k]\n"
    "    return helper(n)\n"
)


def test_identical_code_scores_perfectly():
    assert content_score(ITERATIVE_FIB, ITERATIVE_FIB)["content_score"] == 1.0


def test_renaming_is_not_a_content_change():
    renamed = "def solve(zzz):\n    qq, ww = 0, 1\n    for _ in range(zzz):\n        qq, ww = ww, qq + ww\n    return qq\n"
    assert content_score(ITERATIVE_FIB, renamed)["content_score"] > 0.95


def test_different_algorithm_scores_low():
    assert content_score(ITERATIVE_FIB, RECURSIVE_FIB)["content_score"] < 0.7


def test_canonicalize_is_idempotent():
    once = canonicalize(ITERATIVE_FIB)
    assert canonicalize(once) == once


def test_canonicalize_collapses_loop_idioms():
    for_version = "def solve(n):\n    t = 0\n    for i in range(n):\n        t += i\n    return t\n"
    while_version = "def solve(n):\n    t = 0\n    i = 0\n    while i < n:\n        t = t + i\n        i = i + 1\n    return t\n"
    assert canonicalize(for_version) == canonicalize(while_version)


def test_legit_and_cheat_populations_are_separable():
    """The property the whole cheat detector rests on.

    Every verified legitimate rewrite must score above every deliberate cheat.
    If this fails, the eval cannot distinguish a faithful rewrite from a model
    that ignored its input, and no downstream training result means anything.
    """
    train_path = ROOT / "data" / "pairs" / "train.jsonl"
    if not train_path.exists():
        import pytest

        pytest.skip("run src/transforms/generate_pairs.py first")

    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    legit = [content_score(r["input"].split("\n", 1)[1], r["target"])["content_score"] for r in rows]

    cheats = []
    for alt_path in (ROOT / "problems").glob("*/solution_alt.py"):
        original = (alt_path.parent / "solution.py").read_text(encoding="utf-8")
        cheats.append(content_score(original, alt_path.read_text(encoding="utf-8"))["content_score"])

    assert legit and cheats
    assert min(legit) > max(cheats), f"populations overlap: legit min {min(legit):.3f} <= cheat max {max(cheats):.3f}"
