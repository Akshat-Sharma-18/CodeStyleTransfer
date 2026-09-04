"""Type-annotation stripping transform: annotated (verbose) -> bare (terse).

Fires on essentially every annotated solution, so it is one of the broadest
signals in the transform set.
"""

from __future__ import annotations

import ast


class StripAnnotations(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.returns = None
        for arg in list(node.args.args) + list(node.args.posonlyargs) + list(node.args.kwonlyargs):
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt:
        self.generic_visit(node)
        if node.value is None:
            # `x: int` with no value has no runtime effect; drop it entirely.
            return ast.Pass()
        return ast.Assign(targets=[node.target], value=node.value)


def strip_annotations(source: str) -> str:
    tree = ast.parse(source)
    new_tree = StripAnnotations().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
