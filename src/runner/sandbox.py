"""The correctness gate: run a candidate rewrite's code against a problem's
test suite in a subprocess, and report pass/fail.

This is deliberately minimal for v1 — subprocess isolation with a timeout,
no containerization yet. Tighten sandboxing before running untrusted model
output at scale.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    passed: bool
    stdout: str
    stderr: str
    timed_out: bool


def run_against_tests(candidate_code: str, test_code: str, timeout_s: float = 5.0) -> RunResult:
    """Write candidate + test code to a temp module and run it with pytest."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "solution.py").write_text(candidate_code, encoding="utf-8")
        (tmp_path / "test_solution.py").write_text(test_code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "test_solution.py", "-q"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return RunResult(
                passed=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return RunResult(passed=False, stdout=exc.stdout or "", stderr=exc.stderr or "", timed_out=True)
