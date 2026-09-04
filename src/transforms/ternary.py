"""if/else assignment <-> ternary expression transform.

Only handles the common shape:
    if cond:
        x = a
    else:
        x = b
<-> x = a if cond else b

Anything more complex is left untouched, consistent with Option A's bounded,
honest transform space.
"""

from __future__ import annotations

import ast


class IfElseToTernary(ast.NodeTransformer):
    def visit_If(self, node: ast.If) -> ast.stmt | list[ast.stmt]:
        self.generic_visit(node)

        is_simple_assign_branch = (
            len(node.body) == 1
            and len(node.orelse) == 1
            and isinstance(node.body[0], ast.Assign)
            and isinstance(node.orelse[0], ast.Assign)
            and len(node.body[0].targets) == 1
            and len(node.orelse[0].targets) == 1
            and isinstance(node.body[0].targets[0], ast.Name)
            and isinstance(node.orelse[0].targets[0], ast.Name)
            and node.body[0].targets[0].id == node.orelse[0].targets[0].id
        )
        if not is_simple_assign_branch:
            return node

        target_name = node.body[0].targets[0]
        ternary = ast.IfExp(test=node.test, body=node.body[0].value, orelse=node.orelse[0].value)
        return ast.Assign(targets=[target_name], value=ternary)


def if_else_to_ternary(source: str) -> str:
    tree = ast.parse(source)
    new_tree = IfElseToTernary().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
