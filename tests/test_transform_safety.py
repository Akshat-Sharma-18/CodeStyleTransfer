"""Regression tests for behaviour-breaking transform bugs.

Every case here was found by the correctness gate running the transforms over
real MBPP solutions, not by inspection. Each asserts the transform either
rewrites correctly or declines to fire -- never that it silently changes
behaviour.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transforms.augassign import expand_aug_assign
from transforms.comprehension import expand_comprehension
from transforms.loops import for_range_to_while


def run(source: str, fn_name: str, *args):
    namespace: dict = {}
    exec(source, namespace)
    return namespace[fn_name](*args)


# --- for_range_to_while -------------------------------------------------


def test_countdown_range_keeps_behaviour():
    """`range(n, -1, -1)` counts down; a `while i < -1` condition never runs.

    Found on mbpp_620, where the rewritten loop body was skipped entirely.
    """
    source = "def f(n):\n    out = []\n    for i in range(n - 2, -1, -1):\n        out.append(i)\n    return out\n"
    transformed = for_range_to_while(source)
    assert run(source, "f", 5) == run(transformed, "f", 5) == [3, 2, 1, 0]


def test_continue_is_refused():
    """The increment sits at the end of the body, so `continue` would skip it
    and hang the loop forever."""
    source = "def f(n):\n    t = 0\n    for i in range(n):\n        if i % 2:\n            continue\n        t += i\n    return t\n"
    transformed = for_range_to_while(source)
    assert "for i in range" in transformed, "must decline rather than emit an infinite loop"


def test_continue_in_nested_loop_still_converts_outer():
    source = (
        "def f(n):\n"
        "    t = 0\n"
        "    for i in range(n):\n"
        "        for j in range(n):\n"
        "            if j:\n"
        "                continue\n"
        "            t += 1\n"
        "    return t\n"
    )
    transformed = for_range_to_while(source)
    assert run(source, "f", 4) == run(transformed, "f", 4)


def test_bound_is_evaluated_once():
    """range(len(xs)) fixes the bound up front; `while i < len(xs)` would not."""
    source = "def f(xs):\n    for i in range(len(xs)):\n        xs.append(0)\n    return len(xs)\n"
    transformed = for_range_to_while(source)
    assert run(source, "f", [1, 2, 3]) == run(transformed, "f", [1, 2, 3]) == 6


def test_index_read_after_loop_is_refused():
    """`for` leaves the index at its last value; the `while` rewrite leaves it
    one past the end. Found on mbpp_819, which reads `lists[i + 1]` after the
    loop and so disagreed with the original."""
    source = (
        "def f(xs):\n"
        "    last = None\n"
        "    for i in range(len(xs) - 1):\n"
        "        last = xs[i]\n"
        "    return (last, xs[i + 1])\n"
    )
    transformed = for_range_to_while(source)
    assert "for i in range" in transformed, "index escapes the loop, so conversion must be declined"
    assert run(source, "f", [1, 2, 3]) == run(transformed, "f", [1, 2, 3])


def test_non_escaping_index_still_converts():
    source = "def f(n):\n    t = 0\n    for i in range(n):\n        t = t + i\n    return t\n"
    transformed = for_range_to_while(source)
    assert "while" in transformed
    assert run(source, "f", 5) == run(transformed, "f", 5) == 10


def test_non_literal_step_is_refused():
    source = "def f(n, s):\n    out = []\n    for i in range(0, n, s):\n        out.append(i)\n    return out\n"
    transformed = for_range_to_while(source)
    assert "for i in range" in transformed, "sign of a non-literal step is unknowable"


# --- expand_comprehension ----------------------------------------------


def test_self_referencing_comprehension_is_refused():
    """`xs = [t for t in xs if t]` must not become `xs = []` then iterate `xs`.

    Found on mbpp_966, where the expansion always returned an empty list.
    """
    source = "def f(xs):\n    xs = [t for t in xs if t]\n    return xs\n"
    transformed = expand_comprehension(source)
    assert run(source, "f", [1, 0, 2]) == run(transformed, "f", [1, 0, 2]) == [1, 2]


def test_scope_leak_is_refused():
    """A comprehension has its own scope; a for loop does not. Expanding one
    would clobber an outer binding of the same name."""
    source = "def f(xs):\n    t = 99\n    ys = [t for t in xs]\n    return t\n"
    transformed = expand_comprehension(source)
    assert run(source, "f", [1, 2, 3]) == run(transformed, "f", [1, 2, 3]) == 99


def test_ordinary_comprehension_still_expands():
    source = "def f(xs):\n    out = [y * 2 for y in xs]\n    return out\n"
    transformed = expand_comprehension(source)
    assert "append" in transformed
    assert run(source, "f", [1, 2]) == run(transformed, "f", [1, 2]) == [2, 4]


# --- expand_aug_assign --------------------------------------------------


def test_sequence_augassign_is_refused():
    """`list += tuple` works via __iadd__; `list = list + tuple` is a TypeError.

    Found on mbpp_750.
    """
    source = "def f(a, b):\n    a += b\n    return a\n"
    transformed = expand_aug_assign(source)
    assert run(source, "f", [1], (2,)) == run(transformed, "f", [1], (2,)) == [1, 2]


def test_numeric_accumulator_still_expands():
    source = "def f(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n"
    transformed = expand_aug_assign(source)
    assert "total = total + i" in transformed
    assert run(source, "f", 5) == run(transformed, "f", 5) == 10
