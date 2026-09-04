import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runner.sandbox import run_against_tests


def test_passing_code():
    code = "def solve(x):\n    return x + 1\n"
    tests = "from solution import solve\n\ndef test_it():\n    assert solve(1) == 2\n"
    result = run_against_tests(code, tests)
    assert result.passed


def test_failing_code():
    code = "def solve(x):\n    return x - 1\n"
    tests = "from solution import solve\n\ndef test_it():\n    assert solve(1) == 2\n"
    result = run_against_tests(code, tests)
    assert not result.passed
