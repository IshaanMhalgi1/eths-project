"""
Single-rater consistency checks for the v2 blind annotation.

WHY
---
Annotator 2 is unavailable, so no inter-annotator kappa can be computed and
the v2 qrels have NO reliability estimate. The term-overlap qrels cannot
substitute (§6b: truncated at 5, corpus-dependent, empty for 20/110 queries).

That leaves internal consistency of the one rater's pass as the only evidence
available. This is a WEAKER substitute and cannot be presented as agreement.
It can, however, falsify specific failure modes:

  1. STRAIGHT-LINING - assigning one grade to everything. Would show as ~0
     variance in the per-query grade distribution.
  2. POSITION BIAS - grading by where a card appeared rather than what it
     says. Candidate order is shuffled per (query, annotator), so if judged
     relevance rises with display position, the shuffle failed or was ignored.
  3. ORDER EFFECTS / FATIGUE - relevance falling monotonically through each
     query's candidate list.
  4. TIME-OF-PASS EFFECTS - the UI records no timestamps per judgment, so
     this is NOT testable. Stated as a limitation rather than checked.

None of these substitute for a second rater. Passing them does not make the
qrels reliable; it only rules out specific ways they could be garbage.
"""
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'data')

EXPORT = os.path.join(DATA, 'annotation_v2_annotator1.json')

# Display order is a stable shuffle per (query, annotator). Reconstruct it the
# same way index.html does so "display position" is meaningful.
def mulberry32(a):
    """Exact port of the PRNG in data/annotation/index.html.

    Python's hash() is per-process randomized, so the display order CANNOT be
    reconstructed with it - doing so would analyse a different permutation than
    the one the rater actually saw, and any position effect measured would be
    meaningless. This is a bit-for-bit port of the JavaScript, verified against
    node in the repo's test history.

    Note `a` is advanced but NOT overwritten by the imul result: in JS only the
    local `t` carries that. Clobbering `a` silently desynchronises the stream.
    """
    M32 = 0xFFFFFFFF

    def f():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = ((a ^ (a >> 15)) * (1 | a)) & M32
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & M32)) ^ t
        t &= M32
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return f


def js_seed(s):
    """Exact port of the seed derivation in index.html."""
    seed = 0
    for ch in s:
        seed = (seed * 31 + ord(ch)) & 0xFFFFFFFF
    return seed


def display_order(qid, annotator, docs):
    """Reproduce the exact order the UI rendered for (query, annotator)."""
    seed = js_seed(f"{qid}|{annotator}")
    rand = mulberry32(seed)
    order = list(docs)
    for i in range(len(order) - 1, 0, -1):
        j = int(rand() * (i + 1))
        order[i], order[j] = order[j], order[i]
    return order


def main():
    export = json.load(open(EXPORT))
    queries = export['queries']
    pool = json.load(open(os.path.join(DATA, 'annotation', 'annotation_pools.json')))
    pool_order = {
        q['query_id']: [c['canonical_parent_doc_id'] for c in q['candidates']]
        for q in pool['queries']
    }

    print("=" * 74)
    print("SINGLE-RATER CONSISTENCY CHECKS (annotator 1, v2 blind)")
    print("=" * 74)
    print("These are failure-mode checks, NOT agreement. No kappa exists.\n")

    # ---- 1. straight-lining -------------------------------------------
    print("-" * 74)
    print("1. STRAIGHT-LINING CHECK (per-query grade variance)")
    print("-" * 74)
    print(f"{'query':<7}{'n':>4}{'g0':>5}{'g1':>5}{'g2':>5}{'distinct':>10}  verdict")
    flat = 0
    for q in queries:
        gs = [c['relevance_annotator1'] for c in q['candidates']]
        g0, g1, g2 = gs.count(0), gs.count(1), gs.count(2)
        distinct = len({g for g in gs if g is not None})
        v = "SUSPECT" if distinct <= 1 else ("ok" if distinct >= 2 else "ok")
        if distinct <= 1:
            flat += 1
        print(f"{q['query_id']:<7}{len(gs):>4}{g0:>5}{g1:>5}{g2:>5}{distinct:>10}  {v}")
    print(f"\n  queries with a single repeated grade: {flat}/{len(queries)}")
    print("  (q10 is genuinely uniform - see report; that is a pool property,")
    print("   not necessarily a rating failure, but it carries no information.)")

    # ---- 2 & 3. position bias / order effect --------------------------
    print()
    print("-" * 74)
    print("2/3. DISPLAY-POSITION EFFECT (order is shuffled; position should not matter)")
    print("-" * 74)

    # Raw bands are CONFOUNDED: pool sizes range 7-30, so a 7-candidate query
    # only ever populates early bands while a 30-candidate query populates all
    # of them. Band membership is therefore entangled with query identity. The
    # controlled comparison is WITHIN query: each query's own early half vs
    # its own late half, then averaged across queries.
    pos_rel = defaultdict(list)
    within = []
    for q in queries:
        qid = q['query_id']
        # Pool order is the order in annotation_pools.json; the UI shuffles a
        # copy of it. Use the pool file, not the export, as the base order.
        docs = pool_order[qid]
        order = display_order(qid, 1, docs)
        g = {c['canonical_parent_doc_id']: c['relevance_annotator1']
             for c in q['candidates']}
        seq = [(pos, 1 if g[d] >= 1 else 0)
               for pos, d in enumerate(order, start=1) if d in g]

        for pos, y in seq:
            pos_rel['all'].append(y)
            b = ('1-5' if pos <= 5 else '6-12' if pos <= 12 else '13+')
            pos_rel[b].append(y)

        n = len(seq)
        if n < 4:
            continue
        half = n // 2
        early = [y for _, y in seq[:half]]
        late = [y for _, y in seq[half:]]
        e, l = sum(early) / len(early), sum(late) / len(late)
        within.append((qid, n, 100 * e, 100 * l, 100 * (e - l)))

    print("RAW bands (CONFOUNDED by unequal pool sizes - reference only):")
    for b in ('1-5', '6-12', '13+', 'all'):
        v = pos_rel[b]
        print(f"  positions {b:<6}: {100*sum(v)/len(v):5.1f}% relevant  (n={len(v)})")

    print()
    print("WITHIN-QUERY early-half vs late-half (removes the query confound):")
    print(f"{'query':<7}{'n':>4}{'early%':>9}{'late%':>8}{'delta':>8}")
    for qid, n, e, l, d in within:
        print(f"{qid:<7}{n:>4}{e:>9.1f}{l:>8.1f}{d:>+8.1f}")
    if within:
        mean_d = sum(x[4] for x in within) / len(within)
        n_pos = sum(1 for x in within if x[4] > 0)
        n_eff = len(within)
        # Exact two-sided sign test on the direction of the within-query effect.
        from math import comb
        k = max(n_pos, n_eff - n_pos)
        p_sign = min(1.0, 2 * sum(comb(n_eff, i) for i in range(k, n_eff + 1)) / 2 ** n_eff)
        print(f"\n  mean within-query early-minus-late: {mean_d:+.1f}pp")
        print(f"  queries where early > late: {n_pos}/{n_eff}")
        print(f"  exact sign test (two-sided): p = {p_sign:.3f}")
        if p_sign < 0.05:
            word = 'PRIMACY' if mean_d > 0 else 'RECENCY'
            print(f"  -> {word} bias DETECTED at p<0.05. Within the same query the rater")
            print(f"     graded {'earlier' if mean_d > 0 else 'later'} cards more generously.")
            print("     This is a genuine threat to the labels, not a display artifact.")
        elif abs(mean_d) >= 10:
            print(f"  -> SUGGESTIVE ONLY (p={p_sign:.3f}, not significant at n=10).")
            print(f"     The direction leans {'primacy' if mean_d > 0 else 'recency'}, but")
            print("     10 queries cannot establish it. Treat as an open concern that")
            print("     would be resolved by re-annotating in randomized blocks, not")
            print("     as a demonstrated bias.")
        else:
            print("  -> no material position effect once query is held fixed")

    # ---- notes usage --------------------------------------------------
    print()
    print("-" * 74)
    print("4. NOTES FIELD USAGE (proxy for engagement)")
    print("-" * 74)
    n_notes = sum(1 for q in queries for c in q['candidates']
                  if (c.get('notes_annotator1') or '').strip())
    print(f"  candidates with a note: {n_notes}/{sum(len(q['candidates']) for q in queries)}")
    if n_notes == 0:
        print("  -> no notes written. Not itself a problem for a forced-choice")
        print("     scale, but it removes the qualitative evidence that would")
        print("     otherwise let a second reviewer audit borderline calls.")

    print()
    print("=" * 74)
    print("WHAT THIS DOES NOT ESTABLISH")
    print("=" * 74)
    print("""  Passing these checks does NOT make the v2 qrels reliable. It cannot.
  With one rater there is no estimate of inter-rater reliability, and no
  statistical procedure can manufacture one after the fact.

  A single rater can be consistently wrong in ways none of the checks above
  detect - a stable but mistaken reading of what "relevant" means for these
  queries would pass every one of them.

  These numbers should be cited as "no gross rater failure detected", never
  as agreement, kappa, or validation.""")


if __name__ == '__main__':
    main()
