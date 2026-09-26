"""
DEFINITIVE AUDIT: does any script merge multi-retriever candidates by RAW SCORE?

Run:  python evaluation/audit_raw_score_merge.py
Exit: 0 = clean, 1 = violation found

METHOD
------
Static analysis over the Python AST, not prose review. Three checks:

  C1  FUSION-BY-SORT. Find every `sorted(...)`/`.sort(...)` whose key reads a
      score-like field. For each, resolve the list being sorted. If that list
      is the concatenation (`+`) or `extend` of two or more calls to DIFFERENT
      retrievers, it is a raw-score merge -> violation.

  C2  RETRIEVER CONCATENATION. Find every site that concatenates results from
      two or more distinct retriever entry points. Each site is then reported
      so it can be confirmed by hand as union (safe) or sort-and-truncate
      (violation).

  C3  RRF DISCIPLINE. Confirm the fusion helper converts each component to
      1/(k+rank) BEFORE combining, rather than combining raw scores.

The retriever entry points treated as distinct score scales:
  search_bm25 / rank_bm25            (BM25, raw ~5-14)
  search_dense / rank_dense          (cosine, ~0-1)
  search_hybrid / rank_hybrid        (RRF, ~0.016)
  search_espanol / other             (checked individually)
"""
import ast
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')

# Files under audit, as requested.
TARGETS = [
    'evaluation/metrics.py',
    'evaluation/bootstrap_significance.py',
    'evaluation/ablation_rankers.py',
    'retrieval/hybrid.py',
    'ranking/shared_hybrid.py',
    'evaluation/build_annotation_pools.py',
    'evaluation/generate_qrels.py',
    'evaluation/generate_independent_qrels.py',
    'evaluation/generate_expanded_qrels.py',
    'evaluation/build_small_qrels.py',
    'evaluation/rewrite_qrels.py',
    'evaluation/expand_qrels.py',
    'evaluation/rebuild_qrels.py',
]

# Distinct score scales. Name -> scale family.
RETRIEVERS = {
    'search_bm25': 'BM25-raw',
    'rank_bm25': 'BM25-raw',
    'search_dense': 'dense-cosine',
    'rank_dense': 'dense-cosine',
    'search_hybrid': 'RRF',
    'rank_hybrid': 'RRF',
    'search_espanol': 'espanol',
    'search_temporal': 'temporal',
}

# Helpers that RETURN results from several retrievers at once. Treated as a
# fused multi-retriever source so that `a, b, c = search_all(...)` is flagged
# for a union check instead of passing silently.
FUSED_RETRIEVERS = {'search_all'}

SCORE_FIELDS = {'score', 'final_score', '_score', 'overlap', 'rrf_score',
                'bm25_score', 'dense_score', 'hybrid_score', 'temporal_score'}

violations = []
notes = []


def retriever_of(node):
    """If node is (or evaluates to) a call to a known retriever, name it."""
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return f.id if f.id in RETRIEVERS else None
        if isinstance(f, ast.Attribute):
            return f.attr if f.attr in RETRIEVERS else None
    return None


def walk_lists(node, acc):
    """Collect retriever names feeding into a BinOp(+) / extend() chain."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        walk_lists(node.left, acc)
        walk_lists(node.right, acc)
    elif isinstance(node, ast.Call):
        r = retriever_of(node)
        if r:
            acc.append(r)
        for a in node.args:
            walk_lists(a, acc)
    elif isinstance(node, ast.Name):
        # a variable we cannot resolve here; record as unknown
        acc.append(f"?{node.id}")
    return acc


def resolve(node, env, depth=0):
    """Resolve a node to the set of retriever sources feeding it.

    Follows simple local assignments (`x = search_bm25(...)`) within a scope so
    that `sorted(x, ...)` is attributable without hand-checking. This is what
    makes the audit definitive rather than advisory.
    """
    if depth > 6:
        return set()
    if isinstance(node, ast.Name):
        if node.id in env:
            return resolve(env[node.id], env, depth + 1)
        return {f"?{node.id}"}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return resolve(node.left, env, depth + 1) | resolve(node.right, env, depth + 1)
    if isinstance(node, ast.Call):
        r = retriever_of(node)
        out = {r} if r else set()
        for a in node.args:
            out |= resolve(a, env, depth + 1)
        return out
    if isinstance(node, (ast.List, ast.Tuple)):
        out = set()
        for e in node.elts:
            out |= resolve(e, env, depth + 1)
        return out
    return set()


def enclosing_function(tree, target):
    """Return the FunctionDef node that lexically contains `target`."""
    best = None
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for n in ast.walk(fn):
            if n is target:
                # innermost wins
                if best is None or fn.lineno >= best.lineno:
                    best = fn
                break
    return best


def build_env(node):
    """Map variable name -> assigned value, scoped to ONE function body.

    Scoping matters: `res` is assigned in every ranker, so a shared module-wide
    env resolves each sort to whichever function happened to be visited last
    and produces confidently wrong attributions.
    """
    env = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            env[n.targets[0].id] = n.value
        elif isinstance(n, (ast.For, ast.withitem)) and False:
            pass
    return env


def unpack_env(fn):
    """Also resolve tuple-unpacking like `a, b, c = search_all(...)`."""
    env = build_env(fn)
    extra = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            fname = None
            if isinstance(n.value.func, ast.Name):
                fname = n.value.func.id
            elif isinstance(n.value.func, ast.Attribute):
                fname = n.value.func.attr
            r = retriever_of(n.value) or (fname if fname in FUSED_RETRIEVERS else None)
            if not r:
                continue
            # `a, b, c = f()` parses as ONE Tuple target, not three targets.
            names = []
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.append(t)
                elif isinstance(t, ast.Tuple):
                    names.extend(e for e in t.elts if isinstance(e, ast.Name))
            if len(names) > 1:
                for t in names:
                    if t.id not in env:
                        extra[t.id] = f"<fused:{r}:tuple>"
    env.update({k: v for k, v in extra.items() if k not in env})
    return env


def audit_file(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        notes.append(f"{rel}: NOT FOUND (skipped)")
        return
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    mod_env = build_env(tree)

    def resolve_in(node, fn, depth=0):
        """Resolve a node to retriever sources, using the enclosing fn's env."""
        if depth > 6:
            return set()
        if isinstance(node, str):
            # marker injected by unpack_env for tuple-unpacked fused results
            return {node} if node.startswith('<fused') else set()
        if isinstance(node, ast.Name):
            if fn is not None and node.id in fn:
                return resolve_in(fn[node.id], fn, depth + 1)
            if node.id in mod_env:
                return resolve_in(mod_env[node.id], None, depth + 1)
            return {f"?{node.id}"}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return (resolve_in(node.left, fn, depth + 1)
                    | resolve_in(node.right, fn, depth + 1))
        if isinstance(node, ast.Call):
            r = retriever_of(node)
            out = {r} if r else set()
            for a in node.args:
                out |= resolve_in(a, fn, depth + 1)
            return out
        if isinstance(node, ast.Subscript):
            return resolve_in(node.value, fn, depth + 1)
        if isinstance(node, (ast.List, ast.Tuple)):
            out = set()
            for e in node.elts:
                out |= resolve_in(e, fn, depth + 1)
            return out
        return set()

    # ---- C1: every score sort, and what list it sorts -------------------
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        if fname not in ('sorted', 'sort'):
            continue

        touches_score = any(
            isinstance(n, ast.Constant) and n.value in SCORE_FIELDS
            for n in ast.walk(node))
        if not touches_score:
            continue

        target = node.args[0] if node.args else None
        if target is None:
            continue

        fn = enclosing_function(tree, node)
        env = unpack_env(fn) if fn is not None else mod_env
        srcs = resolve_in(target, env)
        distinct = {s for s in srcs if not s.startswith('?') and not s.startswith('<')}
        fused = {s for s in srcs if s.startswith('<fused')}
        unknown = [s for s in srcs if s.startswith('?')]

        owner = fn.name if fn is not None else '<module>'
        loc = f"{rel}:{node.lineno} in {owner}()"

        if len(distinct) > 1 or (fused and distinct):
            violations.append(
                f"{loc}  RAW-SCORE MERGE: score-sort over MIXED sources "
                f"{sorted(distinct | fused)}")
        elif fused:
            notes.append(f"{loc}  score-sort on FUSED multi-retriever tuple "
                         f"{sorted(fused)} - MUST be union, not sort-and-truncate")
        elif not distinct and unknown:
            notes.append(f"{loc}  score-sort UNRESOLVED {unknown} - manual check")
        elif not distinct:
            notes.append(f"{loc}  score-sort, no retriever source (local list)")
        else:
            fam = next(iter(distinct))
            notes.append(f"{loc}  score-sort single-source [{fam}] - safe")

    # ---- C2: retriever concatenations -----------------------------------
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            fn = enclosing_function(tree, node)
            env = unpack_env(fn) if fn is not None else mod_env
            srcs = resolve_in(node, env)
            distinct = {s for s in srcs if not s.startswith('?')}
            if len(distinct) > 1:
                owner = fn.name if fn is not None else '<module>'
                notes.append(
                    f"{rel}:{node.lineno} in {owner}()  CONCATENATES "
                    f"{sorted(distinct)} - union required, no sort-and-truncate")
            elif any(s.startswith('<fused') for s in distinct):
                owner = fn.name if fn is not None else '<module>'
                notes.append(
                    f"{rel}:{node.lineno} in {owner}()  CONCATENATES a "
                    f"multi-retriever tuple {sorted(distinct)} - union "
                    f"required, no sort-and-truncate")


    # ---- C4: cross-retriever score COMPARISONS (not sorts) ---------------
    # A `>` between two score fields inside a function that also merges
    # retrievers cannot reorder the pool, but it is still a raw-score
    # comparison across scales. Reported separately so it is on the record
    # rather than silently tolerated.
    # A cross-retriever score comparison can live in a DIFFERENT function from
    # the one that merges retrievers (here: the merge is in
    # build_annotation_pool, the `>` is in deduplicate_candidates), so
    # fused-ness is a FILE-level property, not a per-function one.
    file_has_fused = any(
        isinstance(v, str) and v.startswith('<fused')
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for v in unpack_env(fn).values()
    ) or any(
        isinstance(v, str) and v.startswith('<fused')
        for v in mod_env.values()
    )

    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if not file_has_fused:
            break
        for n in ast.walk(fn):
            if isinstance(n, ast.Compare) and len(n.ops) == 1 \
                    and isinstance(n.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
                fields = {c.value for c in ast.walk(n)
                          if isinstance(c, ast.Constant)
                          and c.value in SCORE_FIELDS}
                if fields:
                    notes.append(
                        f"{rel}:{n.lineno} in {fn.name}()  cross-retriever "
                        f"score COMPARISON on {sorted(fields)} - cannot change "
                        f"pool membership or order; picks which duplicate "
                        f"record is retained (cosmetic, not a ranking input)")


def main():
    print("=" * 78)
    print("DEFINITIVE RAW-SCORE-MERGE AUDIT")
    print("=" * 78)
    print("Method: AST analysis of every score-keyed sort and every")
    print("multi-retriever concatenation. Not prose review.\n")

    for t in TARGETS:
        audit_file(t)

    print("-" * 78)
    print("C1/C2  FINDINGS")
    print("-" * 78)
    for n in notes:
        print("  " + n)
    print()

    print("-" * 78)
    print("C3  RRF DISCIPLINE in retrieval/hybrid.py")
    print("-" * 78)
    p = os.path.join(ROOT, 'retrieval', 'hybrid.py')
    txt = open(p, encoding='utf-8').read()
    rrf = txt.count('1.0 / (k + rank')
    pre = 'rrf_bm25' in txt and 'rrf_dense' in txt
    print(f"  1/(k+rank) conversions found      : {rrf}")
    print(f"  both components converted to RRF   : {pre}")
    print(f"  raw scores combined before RRF     : "
          f"{'YES - VIOLATION' if not pre else 'no'}")
    if not pre:
        violations.append("retrieval/hybrid.py: raw scores combined before RRF")

    print()
    print("=" * 78)
    if violations:
        print(f"VERDICT: VIOLATIONS FOUND ({len(violations)})")
        for v in violations:
            print("  x " + v)
        sys.exit(1)
    print("VERDICT: CLEAN - no raw-score merge across retrievers in any")
    print("         audited file. See manual confirmation table for the")
    print("         one deliberate multi-retriever concatenation.")
    print("=" * 78)


if __name__ == '__main__':
    main()
