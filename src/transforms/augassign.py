"""Augmented-assignment expansion: `x += 1` (terse) <-> `x = x + 1` (verbose).

These two are *not* equivalent in general, which the correctness gate caught on
mbpp_750:

    test_list += test_tup    # list.__iadd__ -- extends in place, takes any iterable
    test_list = test_list + test_tup    # list.__add__ -- TypeError for list + tuple

The in-place form is also observable through aliases: `a = b; a += [1]` mutates
`b`, while `a = a + [1]` does not. Python cannot tell us the runtime type
statically, so the transform only fires where there is local evidence that the
target is a number -- it was assigned a numeric literal somewhere in the same
scope. That covers the ordinary accumulator (`total = 0` ... `total += i`) and
declines the sequence cases rather than guessing.

Subscript and attribute targets are refused outright: expanding `a[f()] += 1`
would evaluate the index twice.
"""

from __future__ import annotations

import ast


def _numeric_names(tree: ast.AST) -> set[str]:
    """Names assigned a numeric literal somewhere -- weak but safe evidence."""
    numeric: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, (int, float, complex)) and not isinstance(node.value.value, bool):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        numeric.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, (int, float, complex)) and isinstance(node.target, ast.Name):
                numeric.add(node.target.id)
    return numeric


class ExpandAugAssign(ast.NodeTransformer):
    def __init__(self, numeric_names: set[str]) -> None:
        self.numeric_names = numeric_names

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        self.generic_visit(node)
        if not isinstance(node.target, ast.Name):
            return node
        if node.target.id not in self.numeric_names:
            return node

        load_target = ast.Name(id=node.target.id, ctx=ast.Load())
        expanded = ast.BinOp(left=load_target, op=node.op, right=node.value)
        return ast.Assign(targets=[node.target], value=expanded)


def expand_aug_assign(source: str) -> str:
    tree = ast.parse(source)
    new_tree = ExpandAugAssign(_numeric_names(tree)).visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
