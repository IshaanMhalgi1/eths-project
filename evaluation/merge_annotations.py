"""
Merge two annotator files into final human-annotated qrels.
Computes inter-annotator agreement and resolves disagreements.

Resolution method: MEAN (average of the two annotators' scores, rounded).
Rationale: 3-point scale, symmetric disagreement (e.g. 0 vs 2 is as
contentious as 0 vs 1), and n=10 queries means we cannot afford a third
annotator for tiebreaks. Mean preserves gradation; a "conservative min"
would systematically bias qrels toward Not-Relevent and understate recall
in exactly the lexical-bias direction this exercise exists to correct.

Judgments with mean >= 1.0 are treated as "relevant" for qrels purposes
(standard graded-relevance binarization at the partial threshold), since
metrics.py expects a binary relevant_doc_ids list.

SAFETY: the two files MUST come from the same pool generation. v1 (q5/q21,
158 candidates, retriever scores visible) and v2 (q10/q33, 183 candidates,
blind) are different instruments and cannot be merged. Mixing them silently
produces nonsense. This script now refuses mismatched pools and writes
version-suffixed outputs so it cannot clobber a prior generation.
"""
import json
import os
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'data')

POOL_PATH = os.path.join(DATA, 'annotation', 'annotation_pools.json')


def load(path):
    with open(path) as f:
        return json.load(f)


def pool_fingerprint(path):
    """Identify which annotation generation a file belongs to."""
    with open(path) as f:
        d = json.load(f)
    return {q['query_id'] for q in d['queries']}, sum(
        len(q['candidates']) for q in d['queries'])


def cohen_kappa(pairs):
    """Cohen's kappa for a list of (label1, label2) integer pairs."""
    n = len(pairs)
    if n == 0:
        return None
    cats = sorted({l for p in pairs for l in p})
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}

    # Observed agreement
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n

    # Expected agreement
    ca = [0] * k
    cb = [0] * k
    for a, b in pairs:
        ca[idx[a]] += 1
        cb[idx[b]] += 1
    pe = sum((ca[i] / n) * (cb[i] / n) for i in range(k))

    if pe == 1.0:
        return None  # degenerate: no variance
    return (po - pe) / (1 - pe)


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: merge_annotations.py <annotator1.json> <annotator2.json>")
    A1_PATH, A2_PATH = sys.argv[1], sys.argv[2]

    a1 = load(A1_PATH)
    a2 = load(A2_PATH)

    # ---- pool-generation guard -------------------------------------------
    a1_qids, a1_n = pool_fingerprint(A1_PATH)
    a2_qids, a2_n = pool_fingerprint(A2_PATH)
    try:
        live_qids, live_n = pool_fingerprint(POOL_PATH)
        live_label = f"{len(live_qids)} queries / {live_n} candidates"
    except FileNotFoundError:
        live_qids, live_n, live_label = None, None, "not found"

    print("=" * 70)
    print("POOL GENERATION GUARD")
    print("=" * 70)
    print(f"  annotator 1 : {len(a1_qids)} queries / {a1_n} candidates")
    print(f"  annotator 2 : {len(a2_qids)} queries / {a2_n} candidates")
    print(f"  live pool   : {live_label}")

    if a1_qids != a2_qids or a1_n != a2_n:
        print("\n  ABORT: annotator files come from DIFFERENT pool generations.")
        print(f"    a1-only queries: {sorted(a1_qids - a2_qids)}")
        print(f"    a2-only queries: {sorted(a2_qids - a1_qids)}")
        print("    v1 and v2 pools are different instruments and must not be merged.")
        sys.exit(2)

    if live_qids is not None and a1_qids != live_qids:
        print("\n  ABORT: these files do not match the CURRENT blind pool.")
        print(f"    file-only queries: {sorted(a1_qids - live_qids)}")
        print(f"    pool-only queries: {sorted(live_qids - a1_qids)}")
        print("    Re-run against matching exports, or restore the correct pool.")
        sys.exit(2)

    # ---- self-merge guard -------------------------------------------------
    if os.path.abspath(A1_PATH) == os.path.abspath(A2_PATH):
        sys.exit("ABORT: both arguments are the same file. Merging an annotator "
                 "with themselves yields a meaningless kappa of 1.0. Pass the "
                 "two annotators' separate exports.")
    if a1.get('annotator') == a2.get('annotator'):
        sys.exit(f"ABORT: both files declare annotator={a1.get('annotator')!r}. "
                 "Agreement requires two different annotators.")

    suffix = "v2" if a1_qids == live_qids else "other"
    print(f"  OK: consistent pool, writing suffixed '{suffix}' outputs.\n")

    q1 = {q['query_id']: q for q in a1['queries']}
    q2 = {q['query_id']: q for q in a2['queries']}

    print("=" * 70)
    print("INTER-ANNOTATOR AGREEMENT")
    print("=" * 70)

    all_pairs = []
    disagreements = []
    per_query_stats = {}

    for qid in q1:
        if qid not in q2:
            print(f"  WARNING: {qid} missing from annotator 2 file")
            continue

        c1 = {c['canonical_parent_doc_id']: c for c in q1[qid]['candidates']}
        c2 = {c['canonical_parent_doc_id']: c for c in q2[qid]['candidates']}

        shared = set(c1) & set(c2)
        if len(shared) != len(c1) or len(shared) != len(c2):
            print(f"  WARNING: {qid} candidate mismatch "
                  f"(a1={len(c1)}, a2={len(c2)}, shared={len(shared)})")

        pairs = []
        for doc_id in sorted(shared):
            s1 = c1[doc_id]['relevance_annotator1']
            s2 = c2[doc_id]['relevance_annotator2']
            if s1 is None or s2 is None:
                print(f"  WARNING: {qid}/{doc_id} has null judgment "
                      f"(a1={s1}, a2={s2})")
                continue
            pairs.append((s1, s2))
            all_pairs.append((s1, s2))
            if s1 != s2:
                disagreements.append({
                    'query_id': qid,
                    'query_text': q1[qid]['query_text'],
                    'doc_id': doc_id,
                    'a1': s1, 'a2': s2,
                    'notes1': c1[doc_id].get('notes_annotator1', ''),
                    'notes2': c2[doc_id].get('notes_annotator2', ''),
                })

        agree_n = sum(1 for a, b in pairs if a == b)
        kappa = cohen_kappa(pairs)
        per_query_stats[qid] = {
            'n': len(pairs),
            'agree': agree_n,
            'pct': 100 * agree_n / len(pairs) if pairs else 0,
            'kappa': kappa,
        }
        k_str = f"{kappa:.3f}" if kappa is not None else "n/a"
        print(f"  {qid:>6} ({q1[qid]['query_text'][:32]:<32}): "
              f"{agree_n}/{len(pairs)} agree ({100*agree_n/len(pairs) if pairs else 0:.0f}%), "
              f"kappa={k_str}")

    total = len(all_pairs)
    total_agree = sum(1 for a, b in all_pairs if a == b)
    overall_kappa = cohen_kappa(all_pairs)
    print(f"\n  OVERALL: {total_agree}/{total} agree "
          f"({100*total_agree/total:.1f}%), "
          f"kappa={overall_kappa:.3f}" if overall_kappa is not None else "")

    # Interpretation
    print("\n  Interpretation:")
    if overall_kappa is None:
        print("    kappa undefined (no label variance)")
    elif overall_kappa >= 0.8:
        print("    kappa >= 0.80: strong agreement — qrels are trustworthy")
    elif overall_kappa >= 0.6:
        print("    kappa 0.60-0.80: moderate agreement — usable, report with caveat")
    else:
        print("    kappa < 0.60: weak agreement — qrels reliability is questionable")

    print("\n" + "=" * 70)
    print(f"DISAGREEMENTS ({len(disagreements)} of {total} judgments)")
    print("=" * 70)
    for d in disagreements:
        note_str = ""
        if d['notes1'] or d['notes2']:
            note_str = f"  [notes: a1='{d['notes1']}' a2='{d['notes2']}']"
        print(f"  {d['query_id']:>6} / {d['doc_id']:<14} "
              f"a1={d['a1']} a2={d['a2']}{note_str}")

    # Build final qrels
    print("\n" + "=" * 70)
    print("BUILDING FINAL QRELS")
    print("=" * 70)

    final_queries = []
    per_doc_audit = []

    for qid in q1:
        if qid not in q2:
            continue
        c1 = {c['canonical_parent_doc_id']: c for c in q1[qid]['candidates']}
        c2 = {c['canonical_parent_doc_id']: c for c in q2[qid]['candidates']}
        shared = set(c1) & set(c2)

        relevant = []
        for doc_id in sorted(shared):
            s1 = c1[doc_id]['relevance_annotator1']
            s2 = c2[doc_id]['relevance_annotator2']
            if s1 is None or s2 is None:
                continue
            mean_score = (s1 + s2) / 2
            if mean_score >= 1.0:
                relevant.append(doc_id)
            per_doc_audit.append({
                'query_id': qid,
                'doc_id': doc_id,
                'a1': s1, 'a2': s2,
                'mean': mean_score,
                'binarized': mean_score >= 1.0,
            })

        final_queries.append({
            'query_id': qid,
            'query_text': q1[qid]['query_text'],
            'temporal_type': q1[qid]['temporal_type'],
            'corpus': q1[qid]['corpus'],
            'relevant_doc_ids': relevant,
        })

        n_rel = len(relevant)
        n_cand = len([d for d in shared])
        print(f"  {qid:>6}: {n_rel}/{n_cand} candidates relevant "
              f"({100*n_rel/n_cand if n_cand else 0:.0f}%)")

    output = {
        '_schema_note': 'Human-annotated qrels. relevant_doc_ids = binarized from 3-point '
                        'scale (mean of 2 annotators >= 1.0). Non-lexical ground truth.',
        'resolution_method': 'MEAN of two annotators, binarized at >=1.0',
        'n_queries': len(final_queries),
        'n_judgments': total,
        'percent_agreement': round(100 * total_agree / total, 1) if total else 0,
        'cohen_kappa': round(overall_kappa, 3) if overall_kappa is not None else None,
        'queries': final_queries,
    }

    out_path = os.path.join(DATA, f'qrels_human_annotated_{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved final qrels to {out_path}")

    # Audit file with per-judgment detail
    audit_path = os.path.join(DATA, f'qrels_human_annotated_{suffix}_audit.json')
    with open(audit_path, 'w') as f:
        json.dump({
            '_note': 'Per-judgment annotator labels preserved for agreement auditing.',
            'resolution_method': output['resolution_method'],
            'percent_agreement': output['percent_agreement'],
            'cohen_kappa': output['cohen_kappa'],
            'per_query_stats': per_query_stats,
            'disagreements': disagreements,
            'judgments': per_doc_audit,
        }, f, indent=2)
    print(f"Saved audit trail to {audit_path}")


if __name__ == '__main__':
    main()
