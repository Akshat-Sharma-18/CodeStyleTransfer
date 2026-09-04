"""Augmented-assignment expansion: `x += 1` (terse) <-> `x = x + 1` (verbose).

Only expands when the target is a plain Name, since expanding a subscript or
attribute target (`a[i] += f()`) can change evaluation-count semantics.
"""

from __future__ import annotations

import ast


class ExpandAugAssign(ast.NodeTransformer):
    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt:
        self.generic_visit(node)
        if not isinstance(node.target, ast.Name):
            return node

        load_target = ast.Name(id=node.target.id, ctx=ast.Load())
        expanded = ast.BinOp(left=load_target, op=node.op, right=node.value)
        return ast.Assign(targets=[node.target], value=expanded)


def expand_aug_assign(source: str) -> str:
    tree = ast.parse(source)
    new_tree = ExpandAugAssign().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
