"""Comment/docstring stripping transform: verbose (commented) -> terse (bare)."""

from __future__ import annotations

import ast


def strip_comments_and_docstrings(source: str) -> str:
    """Remove docstrings and re-emit via ast.unparse (drops `#` comments too,
    since those never survive parsing in the first place)."""
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
                if not node.body:
                    node.body.append(ast.Pass())

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
