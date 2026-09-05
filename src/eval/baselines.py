"""Baseline rewriters, scored by the same harness as any trained model.

Two of these exist to prove the eval itself works:

  identity        -- changes nothing. Scores 100% functional correctness and a
                     perfect content score while doing literally no work. If
                     your headline number is correctness alone, this "wins".

  canonical_cheat -- ignores its input and emits a different algorithm that
                     happens to pass the same tests. Also scores 100%
                     correctness. This is the failure mode the spec warns
                     about, and the content metric has to catch it.

If the eval table cannot separate these two from a real rewriter, the eval is
not measuring anything and no training result on top of it is trustworthy.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from transforms.annotations import strip_annotations  # noqa: E402
from transforms.augassign import expand_aug_assign  # noqa: E402
from transforms.comments import strip_comments_and_docstrings  # noqa: E402
from transforms.comprehension import expand_comprehension  # noqa: E402
from transforms.rename import terse_rename  # noqa: E402


def identity(code: str, direction: str) -> str:
    """Do nothing. The null baseline."""
    return code


def normalize_only(code: str, direction: str) -> str:
    """A formatter-style baseline: reformat, change no style decisions.

    This is the "would a linter have done just as well?" comparison the spec
    asks for.
    """
    return ast.unparse(ast.parse(code))


def _load_alternates() -> dict[str, str]:
    alternates = {}
    for alt_path in sorted((ROOT / "problems").glob("*/solution_alt.py")):
        alternates[alt_path.parent.name] = alt_path.read_text(encoding="utf-8")
    return alternates


_ALTERNATES = _load_alternates()


def make_canonical_cheat(problem: str):
    """A rewriter that ignores its input entirely and emits a memorized,
    behaviourally-equivalent but structurally different solution."""

    def cheat(code: str, direction: str) -> str:
        return _ALTERNATES.get(problem, code)

    return cheat


def rule_based_rewriter(code: str, direction: str) -> str:
    """An honest non-learned rewriter: apply the mechanical transforms directly.

    This is the bar a fine-tuned model has to clear to justify training at all,
    so it is worth making it as strong as the transform set allows rather than
    leaving it a straw man.

    The two directions are asymmetric, and watching that asymmetry move with
    the corpus is the clearest evidence in the project that per-direction
    reporting is worth the trouble.

    On the MBPP-only corpus this baseline scored 15% going terse against 36%
    going verbose -- backwards from the obvious story, in which terse is easy
    (delete docstrings and annotations, shorten names) and verbose is hard
    (the added information is not recoverable from the code). The diagnosis was
    that this said nothing about the task and everything about the data: 0.0%
    of MBPP solutions carry a docstring or a type annotation, so two of the
    three terse-direction transforms had nothing to delete and only renaming
    fired, which on its own rarely flips a majority of the style proxies.

    Adding HumanEval (99.4% docstrings, 36.6% annotations) tested that
    prediction. Terse went 15% -> 45% and the ordering flipped back to the
    intuitive one, 45% terse against 22% verbose.

    So the headline style number is partly a statement about the corpus, and a
    single averaged figure would have hidden the whole effect.
    """
    if direction == "to_terse":
        code = strip_comments_and_docstrings(code)
        code = strip_annotations(code)
        return terse_rename(code).code

    # to_verbose: structural expansion only -- no way to invent names or docs.
    for transform in (expand_comprehension, expand_aug_assign):
        try:
            code = transform(code)
        except Exception:  # noqa: BLE001 - a transform that cannot handle this code
            pass
    return ast.unparse(ast.parse(code))


BASELINES = {
    "identity": identity,
    "normalize_only": normalize_only,
    "rule_based": rule_based_rewriter,
}
