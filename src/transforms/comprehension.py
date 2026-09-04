"""List-comprehension expansion: `xs = [f(y) for y in ys]` (terse)
<-> an explicit accumulate loop (verbose).

Only handles a single-generator comprehension bound directly to a plain Name,
optionally with one `if` filter. Nested generators and comprehensions used
inline as sub-expressions are left alone.
"""

from __future__ import annotations

import ast


class ExpandComprehension(ast.NodeTransformer):
    def visit_Assign(self, node: ast.Assign) -> ast.stmt | list[ast.stmt]:
        self.generic_visit(node)

        is_simple_listcomp_assign = (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.ListComp)
            and len(node.value.generators) == 1
            and len(node.value.generators[0].ifs) <= 1
            and not node.value.generators[0].is_async
        )
        if not is_simple_listcomp_assign:
            return node

        target_name = node.targets[0].id
        comp = node.value
        generator = comp.generators[0]

        init = ast.Assign(
            targets=[ast.Name(id=target_name, ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        )
        append_call = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=target_name, ctx=ast.Load()),
                    attr="append",
                    ctx=ast.Load(),
                ),
                args=[comp.elt],
                keywords=[],
            )
        )

        loop_body: list[ast.stmt] = [append_call]
        if generator.ifs:
            loop_body = [ast.If(test=generator.ifs[0], body=loop_body, orelse=[])]

        loop = ast.For(target=generator.target, iter=generator.iter, body=loop_body, orelse=[])
        return [init, loop]


def expand_comprehension(source: str) -> str:
    tree = ast.parse(source)
    new_tree = ExpandComprehension().visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
