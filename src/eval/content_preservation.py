"""Content-preservation metrics: did the rewrite keep the *original's* logic,
or did it just emit some other correct program?

This is the guard against the spec's headline failure mode -- a model that
passes the test gate by reproducing a memorized canonical solution while
ignoring its input. "Passes tests" is necessary but nowhere near sufficient.
"""

from __future__ import annotations

import ast
from collections import Counter
from difflib import SequenceMatcher


def _node_multiset(source: str) -> Counter:
    """Multiset of AST node types, ignoring identifier names.

    Names are deliberately excluded: renaming is a legitimate style move, so a
    content metric that punished it would fight the task itself.
    """
    tree = ast.parse(source)
    counts: Counter = Counter()
    for node in ast.walk(tree):
        counts[type(node).__name__] += 1
    return counts


def ast_similarity(source_a: str, source_b: str) -> float:
    """Cosine-like overlap of AST node-type multisets, in [0, 1]."""
    counts_a = _node_multiset(source_a)
    counts_b = _node_multiset(source_b)

    shared = sum((counts_a & counts_b).values())
    total = max(sum(counts_a.values()), sum(counts_b.values()))
    return shared / total if total else 1.0


def _literal_multiset(source: str) -> Counter:
    """Constants appearing in the code -- the strongest signal that two programs
    solve the *same* problem instance rather than merely similar shapes."""
    tree = ast.parse(source)
    counts: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
            counts[repr(node.value)] += 1
    return counts


def literal_overlap(source_a: str, source_b: str) -> float:
    counts_a = _literal_multiset(source_a)
    counts_b = _literal_multiset(source_b)
    if not counts_a and not counts_b:
        return 1.0

    shared = sum((counts_a & counts_b).values())
    total = max(sum(counts_a.values()), sum(counts_b.values()))
    return shared / total if total else 1.0


def structural_similarity(source_a: str, source_b: str) -> float:
    """difflib ratio over the normalized (unparsed) token streams."""
    normalized_a = ast.unparse(ast.parse(source_a))
    normalized_b = ast.unparse(ast.parse(source_b))
    return SequenceMatcher(None, normalized_a.split(), normalized_b.split()).ratio()


CONTROL_FLOW_NODES = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try, ast.With, ast.FunctionDef)


def _control_flow_paths(source: str) -> Counter:
    """Multiset of nesting paths through control-flow nodes, e.g. 'For>For>If'.

    A plain node-type bag cannot tell a nested double loop from a single loop --
    both are made of the same node types. Nesting *paths* can, which is exactly
    the difference between an O(n^2) brute force and a linear scan.
    """
    tree = ast.parse(source)
    paths: Counter = Counter()

    def walk(node: ast.AST, prefix: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, CONTROL_FLOW_NODES):
                new_prefix = prefix + (type(child).__name__,)
                paths[">".join(new_prefix)] += 1
                walk(child, new_prefix)
            else:
                walk(child, prefix)

    walk(tree, ())
    return paths


def control_flow_similarity(source_a: str, source_b: str) -> float:
    paths_a = _control_flow_paths(source_a)
    paths_b = _control_flow_paths(source_b)
    if not paths_a and not paths_b:
        return 1.0

    shared = sum((paths_a & paths_b).values())
    total = max(sum(paths_a.values()), sum(paths_b.values()))
    return shared / total if total else 1.0


def content_score(original: str, rewrite: str, canonical: bool = True) -> dict[str, float]:
    """Combined content-preservation score and its components.

    By default both sides are first reduced to a style-normal form, so that
    style differences the transforms can undo do not read as content changes.
    Pass canonical=False to score the raw sources -- useful for showing *why*
    the canonical version is needed, since on raw sources the legitimate and
    cheating populations overlap.

    Caveat worth keeping in the writeup: the control-flow term was added
    *because* the brute-force max_subarray cheat slipped past a node-bag-only
    score. It is fitted to a known failure, not independently validated.
    """
    if canonical:
        from eval.canonicalize import canonicalize

        try:
            original = canonicalize(original)
            rewrite = canonicalize(rewrite)
        except SyntaxError:
            pass

    ast_sim = ast_similarity(original, rewrite)
    literal_sim = literal_overlap(original, rewrite)
    struct_sim = structural_similarity(original, rewrite)
    flow_sim = control_flow_similarity(original, rewrite)
    combined = 0.3 * ast_sim + 0.2 * literal_sim + 0.2 * struct_sim + 0.3 * flow_sim

    return {
        "ast_similarity": ast_sim,
        "literal_overlap": literal_sim,
        "structural_similarity": struct_sim,
        "control_flow_similarity": flow_sim,
        "content_score": combined,
    }
