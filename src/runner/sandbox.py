"""The correctness gate: run a candidate rewrite's code against a problem's
test suite in a subprocess, and report pass/fail.

This is deliberately minimal for v1 -- subprocess isolation with a timeout,
no containerization yet. Tighten sandboxing before running untrusted model
output at scale.

On timeouts: the timeout exists to catch genuine infinite loops, not to
measure speed. Under parallel verification (12 workers on 8 cores) a
legitimately slow-but-correct solution can be starved of CPU and blow a tight
deadline -- mbpp_123 runs in 3.4s alone and was being dropped under load at a
5s limit. That silently deleted valid training pairs and made eval results
depend on machine load, so a timeout is now retried once, serially and with a
longer deadline, before it counts as a failure. Timeouts are also reported
separately from real test failures: they mean something different.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 15.0
TIMEOUT_RETRY_MULTIPLIER = 3.0


@dataclass
class RunResult:
    passed: bool
    stdout: str
    stderr: str
    timed_out: bool


def _run_once(candidate_code: str, test_code: str, timeout_s: float) -> RunResult:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "solution.py").write_text(candidate_code, encoding="utf-8")
        (tmp_path / "test_solution.py").write_text(test_code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "test_solution.py", "-q", "-p", "no:cacheprovider"],
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
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return RunResult(passed=False, stdout=stdout, stderr=stderr, timed_out=True)


def run_against_tests(
    candidate_code: str,
    test_code: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    retry_on_timeout: bool = True,
) -> RunResult:
    """Write candidate + test code to a temp module and run it with pytest.

    A timeout is retried once with a longer deadline, so CPU contention during
    parallel verification does not masquerade as a correctness failure.
    """
    result = _run_once(candidate_code, test_code, timeout_s)
    if result.timed_out and retry_on_timeout:
        result = _run_once(candidate_code, test_code, timeout_s * TIMEOUT_RETRY_MULTIPLIER)
    return result
