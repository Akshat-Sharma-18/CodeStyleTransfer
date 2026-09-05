"""for-range <-> while loop-idiom transform.

Only handles the common `for i in range(...)` shape; anything else is left
untouched, since the point is a bounded, honest transform space (Option A),
not a general Python rewriter.

Getting this right is harder than it looks, and the correctness gate caught
every one of these the hard way (32 failures across MBPP before the fixes):

  * **Step sign.** `range(n, -1, -1)` counts down, so the loop condition must
    be `i > stop`, not `i < stop`. Emitting `<` unconditionally turned every
    countdown loop into a no-op. The step's sign is only knowable statically
    when it is a literal, so a non-literal step is refused outright.
  * **`continue`.** The increment lives at the end of the rewritten body, so a
    `continue` jumps straight past it -- turning the loop infinite. Refused.
  * **Single evaluation of the bound.** `range(len(xs))` evaluates `len(xs)`
    exactly once; a `while i < len(xs)` re-evaluates it every iteration, which
    differs whenever the body mutates `xs`. The bound is hoisted into a fresh
    variable to preserve range's semantics.

  * **The index after the loop.** A `for` leaves the index at its last value;
    the rewritten `while` leaves it one step past the end. mbpp_819 reads
    `lists[i + 1]` after its loop, so the two disagree. Rather than attempt
    real flow analysis, any loop whose index is read outside its own body is
    refused -- conservative, and it costs only the loops where the rewrite
    would have been wrong anyway.

What remains unhandled: a `range` whose bound is a non-literal expression that
is *also* mutated by the body is hoisted correctly, but an empty range still
binds the index where `for` would leave it unbound. That case needs the index
to be read after an empty loop, which the refusal above already covers.
"""

from __future__ import annotations

import ast


def _contains_continue_at_this_level(body: list[ast.stmt]) -> bool:
    """True if a `continue` would bind to *this* loop rather than a nested one."""
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Continue):
                # Walk again to see whether an intervening loop captures it.
                if not _is_inside_nested_loop(statement, node):
                    return True
    return False


def _is_inside_nested_loop(root: ast.stmt, target: ast.Continue) -> bool:
    """Is `target` enclosed by a loop that is itself inside `root`?"""

    def search(node: ast.AST, inside_loop: bool) -> bool:
        if node is target:
            return inside_loop
        for child in ast.iter_child_nodes(node):
            child_inside = inside_loop or isinstance(child, (ast.For, ast.AsyncFor, ast.While))
            if search(child, child_inside):
                return True
        return False

    return search(root, isinstance(root, (ast.For, ast.AsyncFor, ast.While)))


def _is_simple_bound(node: ast.expr) -> bool:
    """Cheap to re-evaluate and free of side effects, so no hoist is needed."""
    return isinstance(node, (ast.Constant, ast.Name))


def _loops_whose_index_escapes(tree: ast.AST) -> set[int]:
    """ids of For nodes whose loop variable is read outside their own body.

    Converting those changes the index's value after the loop, so they are
    refused. Reading it before the loop counts too -- rare, and refusing is
    cheaper than proving it safe.
    """
    escaping: set[int] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue

        loop_var = node.target.id
        inside = {id(n) for n in ast.walk(node)}

        for other in ast.walk(tree):
            if (
                isinstance(other, ast.Name)
                and other.id == loop_var
                and isinstance(other.ctx, ast.Load)
                and id(other) not in inside
            ):
                escaping.add(id(node))
                break

    return escaping


class ForRangeToWhile(ast.NodeTransformer):
    def __init__(self, escaping_loops: set[int] | None = None) -> None:
        self.used_names: set[str] = set()
        self.escaping_loops = escaping_loops or set()
        self._counter = 0

    def _fresh_name(self, hint: str) -> str:
        while True:
            candidate = f"_{hint}_{self._counter}"
            self._counter += 1
            if candidate not in self.used_names:
                self.used_names.add(candidate)
                return candidate

    def visit_For(self, node: ast.For) -> ast.stmt | list[ast.stmt]:
        # Checked before descending: generic_visit rewrites nested loops but
        # leaves this node's identity intact, which the precomputed set keys on.
        escapes = id(node) in self.escaping_loops
        self.generic_visit(node)

        if escapes:
            return node

        is_simple_range_loop = (
            isinstance(node.target, ast.Name)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and not node.iter.keywords
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

        # The comparison direction depends on the sign of the step, which we
        # can only know when it is a literal int.
        if not (isinstance(step, ast.Constant) and isinstance(step.value, int) and not isinstance(step.value, bool)):
            if not (
                isinstance(step, ast.UnaryOp)
                and isinstance(step.op, ast.USub)
                and isinstance(step.operand, ast.Constant)
                and isinstance(step.operand.value, int)
            ):
                return node
            step_value = -step.operand.value
        else:
            step_value = step.value

        if step_value == 0:
            return node

        # A `continue` would skip the increment and hang the loop.
        if _contains_continue_at_this_level(node.body):
            return node

        prelude: list[ast.stmt] = []

        # range() evaluates its bound once; `while` would re-evaluate it.
        if _is_simple_bound(stop):
            stop_expr: ast.expr = stop
        else:
            bound_name = self._fresh_name("stop")
            prelude.append(ast.Assign(targets=[ast.Name(id=bound_name, ctx=ast.Store())], value=stop))
            stop_expr = ast.Name(id=bound_name, ctx=ast.Load())

        loop_var = node.target.id
        prelude.append(ast.Assign(targets=[ast.Name(id=loop_var, ctx=ast.Store())], value=start))

        comparison = ast.Lt() if step_value > 0 else ast.Gt()
        condition = ast.Compare(
            left=ast.Name(id=loop_var, ctx=ast.Load()),
            ops=[comparison],
            comparators=[stop_expr],
        )
        increment = ast.AugAssign(
            target=ast.Name(id=loop_var, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(step_value),
        )
        while_node = ast.While(test=condition, body=node.body + [increment], orelse=[])
        return prelude + [while_node]


def for_range_to_while(source: str) -> str:
    tree = ast.parse(source)
    transformer = ForRangeToWhile(_loops_whose_index_escapes(tree))
    transformer.used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)
