"""Identifier renaming transform: descriptive names <-> short names (a, b, x, ...).

Used to generate Option A mechanical training pairs (verbose <-> terse).
"""

from __future__ import annotations

import ast
import string
from dataclasses import dataclass


@dataclass
class RenameResult:
    code: str
    mapping: dict[str, str]


def _short_names(n: int) -> list[str]:
    letters = string.ascii_lowercase
    names, i = [], 0
    while len(names) < n:
        name = letters[i % 26] * (i // 26 + 1)
        names.append(name)
        i += 1
    return names


class _LocalNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for arg in node.args.args:
            self.names.add(arg.arg)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store,)):
            self.names.add(node.id)
        self.generic_visit(node)


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node


def terse_rename(source: str) -> RenameResult:
    """Rewrite local identifiers to short single/double-letter names."""
    tree = ast.parse(source)
    collector = _LocalNameCollector()
    collector.visit(tree)

    builtins_and_keywords = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    candidates = sorted(n for n in collector.names if n not in builtins_and_keywords)

    short = _short_names(len(candidates))
    mapping = dict(zip(candidates, short))

    renamed_tree = _Renamer(mapping).visit(tree)
    ast.fix_missing_locations(renamed_tree)
    return RenameResult(code=ast.unparse(renamed_tree), mapping=mapping)
