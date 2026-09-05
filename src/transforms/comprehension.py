"""List-comprehension expansion: `xs = [f(y) for y in ys]` (terse)
<-> an explicit accumulate loop (verbose).

Only handles a single-generator comprehension bound directly to a plain Name,
optionally with one `if` filter. Nested generators and comprehensions used
inline as sub-expressions are left alone.

Two traps the correctness gate caught, both refused rather than worked around:

  * **Self-reference.** `xs = [t for t in xs if t]` cannot become
    `xs = []` followed by `for t in xs:` -- the initialiser destroys the very
    list being iterated, and the result is always empty. Any comprehension
    that mentions its own assignment target is left alone.
  * **Scope leak.** A comprehension gets its own scope; a `for` loop does not.
    Expanding one leaks the loop variable into the enclosing function, which
    is observable if that name is bound elsewhere. Refused in that case.
"""

from __future__ import annotations

import ast


def _names_in(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _target_names(target: ast.expr) -> set[str]:
    return {child.id for child in ast.walk(target) if isinstance(child, ast.Name)}


class ExpandComprehension(ast.NodeTransformer):
    def __init__(self, bound_elsewhere: set[str] | None = None) -> None:
        self.bound_elsewhere = bound_elsewhere or set()

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

        # Self-reference: initialising the target would destroy the source.
        if target_name in _names_in(comp):
            return node

        # Scope leak: the comprehension variable would escape into the
        # enclosing scope and clobber a binding that is used elsewhere.
        if _target_names(generator.target) & self.bound_elsewhere:
            return node

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


def _names_bound_outside_comprehensions(tree: ast.AST) -> set[str]:
    """Names assigned or bound as parameters anywhere outside a comprehension."""
    comprehension_targets = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in node.generators:
                comprehension_targets.update(id(n) for n in ast.walk(generator.target))

    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and id(node) not in comprehension_targets:
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return bound


def expand_comprehension(source: str) -> str:
    tree = ast.parse(source)
    new_tree = ExpandComprehension(_names_bound_outside_comprehensions(tree)).visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
