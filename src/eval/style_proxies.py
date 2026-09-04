"""Cheap, interpretable style proxies for the verbose <-> terse axis.

These are the "style-target hit" half of the eval table. Deliberately several
of them rather than one: the spec's own warning is that scoring terseness by a
single number (line count) teaches a model to jam everything onto one line.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import asdict, dataclass


@dataclass
class StyleProfile:
    line_count: int
    token_count: int
    mean_identifier_length: float
    comment_ratio: float
    docstring_count: int
    annotation_count: int
    max_line_length: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _identifier_lengths(tree: ast.AST) -> list[int]:
    lengths = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            lengths.append(len(node.id))
        elif isinstance(node, ast.arg):
            lengths.append(len(node.arg))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lengths.append(len(node.name))
    return lengths


def _count_comment_and_total_tokens(source: str) -> tuple[int, int]:
    comment_tokens = 0
    total_tokens = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
                continue
            total_tokens += 1
            if token.type == tokenize.COMMENT:
                comment_tokens += 1
    except (tokenize.TokenError, IndentationError):
        pass
    return comment_tokens, total_tokens


def _count_docstrings(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                count += 1
    return count


def _count_annotations(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            count += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
            count += 1
        elif isinstance(node, ast.AnnAssign):
            count += 1
    return count


def profile(source: str) -> StyleProfile:
    """Compute the style profile of a source string."""
    tree = ast.parse(source)
    lines = [line for line in source.splitlines() if line.strip()]
    identifier_lengths = _identifier_lengths(tree)
    comment_tokens, total_tokens = _count_comment_and_total_tokens(source)

    return StyleProfile(
        line_count=len(lines),
        token_count=total_tokens,
        mean_identifier_length=(sum(identifier_lengths) / len(identifier_lengths)) if identifier_lengths else 0.0,
        comment_ratio=(comment_tokens / total_tokens) if total_tokens else 0.0,
        docstring_count=_count_docstrings(tree),
        annotation_count=_count_annotations(tree),
        max_line_length=max((len(line) for line in lines), default=0),
    )


# Which direction each proxy moves when code becomes MORE terse.
# -1 means "terse code has a lower value for this proxy".
TERSE_DIRECTION = {
    "line_count": -1,
    "token_count": -1,
    "mean_identifier_length": -1,
    "comment_ratio": -1,
    "docstring_count": -1,
    "annotation_count": -1,
    "max_line_length": +1,  # terse code tends to pack lines longer
}


def terseness_shift(verbose_source: str, terse_source: str) -> dict[str, float]:
    """Signed per-proxy deltas, oriented so positive = 'moved toward terse'."""
    verbose_profile = profile(verbose_source).as_dict()
    terse_profile = profile(terse_source).as_dict()

    shift = {}
    for key, direction in TERSE_DIRECTION.items():
        shift[key] = (terse_profile[key] - verbose_profile[key]) * direction
    return shift


def moved_toward_terse(verbose_source: str, terse_source: str) -> bool:
    """True if a majority of the non-zero proxies moved in the terse direction.

    Majority-of-proxies rather than any single number, so a model cannot win by
    gaming one metric.
    """
    shift = terseness_shift(verbose_source, terse_source)
    non_zero = [value for value in shift.values() if value != 0]
    if not non_zero:
        return False
    return sum(1 for value in non_zero if value > 0) > len(non_zero) / 2
