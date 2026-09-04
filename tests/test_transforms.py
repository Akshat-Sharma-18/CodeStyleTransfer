import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transforms.comments import strip_comments_and_docstrings
from transforms.loops import for_range_to_while
from transforms.rename import terse_rename


def test_strip_comments_removes_docstring():
    source = 'def f(x):\n    """docstring"""\n    return x\n'
    result = strip_comments_and_docstrings(source)
    assert "docstring" not in result
    assert "return x" in result


def test_for_range_to_while_preserves_behavior():
    source = "def f():\n    total = 0\n    for i in range(5):\n        total += i\n    return total\n"
    transformed = for_range_to_while(source)
    namespace: dict = {}
    exec(transformed, namespace)
    assert namespace["f"]() == 10


def test_terse_rename_produces_short_names():
    source = "def solve(numbers):\n    total = 0\n    for value in numbers:\n        total += value\n    return total\n"
    result = terse_rename(source)
    assert result.mapping
    assert all(len(short) <= 2 for short in result.mapping.values())
    namespace: dict = {}
    exec(result.code, namespace)
    assert namespace["solve"]([1, 2, 3]) == 6
