"""Map code to a style-normal form, so content can be compared independently
of style.

Why this exists: measuring content preservation directly on the raw sources
does not work. Measured that way, the most aggressive *legitimate* rewrite in
our own training data scores 0.519 while the hardest deliberate cheat scores
0.555 -- the classes overlap, and no threshold separates them. The metric was
conflating "style changed a lot" with "content changed".

The fix is invariance by construction: push both sides through every style
transform we know about, then compare. Anything the style transforms can undo
is style; whatever difference survives is content.

Note the honest limit of this approach -- it is only invariant to the style
axes we implemented. A style move outside that set still shows up as a content
difference.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transforms.annotations import strip_annotations  # noqa: E402
from transforms.augassign import expand_aug_assign  # noqa: E402
from transforms.comments import strip_comments_and_docstrings  # noqa: E402
from transforms.comprehension import expand_comprehension  # noqa: E402
from transforms.loops import for_range_to_while  # noqa: E402
from transforms.ternary import if_else_to_ternary  # noqa: E402


class _CanonicalRenamer(ast.NodeTransformer):
    """Rename every local identifier to v0, v1, ... in first-appearance order.

    Makes the comparison invariant to naming, which is a pure style axis.
    Module-level function names are kept: they are part of the contract, not
    the style.
    """

    def __init__(self, keep: set[str]) -> None:
        self.keep = keep
        self.mapping: dict[str, str] = {}

    def _canonical(self, name: str) -> str:
        if name in self.keep:
            return name
        if name not in self.mapping:
            self.mapping[name] = f"v{len(self.mapping)}"
        return self.mapping[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._canonical(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._canonical(node.arg)
        return node


def _builtin_names() -> set[str]:
    import builtins

    return set(dir(builtins))


def canonicalize(source: str) -> str:
    """Reduce source to a style-normal form.

    Note the direction of the loop and branch transforms. Recovering a `for`
    loop from an arbitrary `while` is not safely decidable, so we do not try.
    Instead we apply the *easy* direction to both sides: a `for i in range(...)`
    loop and its hand-written `while` equivalent both reduce to the same while
    form, which is all invariance actually requires. Same trick for
    if/else-assignment vs ternary.
    """
    # Order matters, and it is load-bearing. Transforms that *generate* nodes
    # must run before the transforms that normalize those node types, or the
    # generated nodes escape normalization and canonicalize() stops being
    # idempotent (a for->while conversion emits an AugAssign for the loop
    # increment; a comprehension expansion can emit a range-for). Producers
    # first, normalizers last.
    code = source
    for transform in (
        strip_comments_and_docstrings,
        strip_annotations,
        expand_comprehension,  # may emit a range-for
        for_range_to_while,  # consumes range-fors, may emit an AugAssign
        if_else_to_ternary,
        expand_aug_assign,  # consumes every AugAssign, generated or original
    ):
        try:
            code = transform(code)
        except Exception:  # noqa: BLE001 - fall through with whatever we have
            pass

    tree = ast.parse(code)
    keep = _builtin_names()
    # Keep top-level function names: they are the callable contract.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            keep.add(node.name)

    renamed = _CanonicalRenamer(keep).visit(tree)
    ast.fix_missing_locations(renamed)
    return ast.unparse(renamed)
