"""for-range <-> while loop-idiom transform.

Only handles the common `for i in range(...)` shape; anything else is left
untouched, since the point is a bounded, honest transform space (Option A),
not a general Python rewriter.
"""

from __future__ import annotations

import ast


class ForRangeToWhile(ast.NodeTransformer):
    def visit_For(self, node: ast.For) -> ast.stmt | list[ast.stmt]:
        self.generic_visit(node)

        is_simple_range_loop = (
            isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and not node.orelse
        )
        if not is_simple_range_loop:
            return node

        args = node.iter.args
        if len(args) == 1:
            start, stop, step = ast.Constant(0), args[0], ast.Constant(1)
        elif len(args) == 2:
            start, stop, step = args[0], args[1], ast.Constant(1)
        elif len(args) == 3:
            start, stop, step = args
        else:
            return node

        loop_var = node.target.id
        init = ast.Assign(targets=[ast.Name(id=loop_var, ctx=ast.Store())], value=start)
        condition = ast.Compare(
            left=ast.Name(id=loop_var, ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[stop],
        )
        increment = ast.AugAssign(
            target=ast.Name(id=loop_var, ctx=ast.Store()),
            op=ast.Add(),
            value=step,
        )
        while_node = ast.While(test=condition, body=node.body + [increment], orelse=[])
        return [init, while_node]


def for_range_to_while(source: str) -> str:
    tree = ast.parse(source)
    new_tree = ForRangeToWhile().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
