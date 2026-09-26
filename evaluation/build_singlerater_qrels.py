"""
Build single-rater human qrels from one annotator's blind export.

WHY THIS EXISTS
---------------
Annotator 2 is unavailable, so the v2 re-annotation will never produce an
inter-annotator kappa. This builds the qrels that CAN be produced from a
single rater, in the same schema the existing analysis scripts consume, so
retrieval metrics can be reported without fabricating a reliability number.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not invent a second annotator, impute agreement, or produce a kappa.
A one-rater qrels set has no reliability estimate attached to it, and every
downstream report must say so.

Graded labels are preserved verbatim (0/1/2). `mean` carries the single
rater's grade so the graded-metric code path (nDCG@10, MAP) works unchanged.
`relevant_doc_ids` is the >=1 binarization, used only where a binary set is
structurally required.
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'data')

EXPORT = os.path.join(DATA, 'annotation_v2_annotator1.json')
POOL = os.path.join(DATA, 'annotation', 'annotation_pools.json')
OUT_QRELS = os.path.join(DATA, 'qrels_human_annotated_v2_singlerater.json')
OUT_AUDIT = os.path.join(DATA, 'qrels_human_annotated_v2_singlerater_audit.json')


def main():
    export = json.load(open(EXPORT))
    pool = {q['query_id']: q for q in json.load(open(POOL))['queries']}

    queries, judgments, per_query = [], [], {}

    for q in export['queries']:
        qid = q['query_id']
        pq = pool.get(qid)
        if pq is None:
            print(f"  WARNING: {qid} not in pool, skipping")
            continue

        relevant, grades, n = [], {}, 0
        for c in q['candidates']:
            g = c.get('relevance_annotator1')
            if g is None:
                continue
            n += 1
            doc = c['canonical_parent_doc_id']
            grades[doc] = g
            judgments.append({
                'query_id': qid,
                'doc_id': doc,
                'a1': g,
                'a2': None,
                'mean': float(g),          # single rater: mean IS the grade
                'binarized': g >= 1,
                'rater': 'annotator1_only',
            })
            if g >= 1:
                relevant.append(doc)

        per_query[qid] = {
            'n': n,
            'g0': sum(1 for v in grades.values() if v == 0),
            'g1': sum(1 for v in grades.values() if v == 1),
            'g2': sum(1 for v in grades.values() if v == 2),
        }
        queries.append({
            'query_id': qid,
            'query_text': q['query_text'],
            'temporal_type': pq.get('temporal_type'),
            'corpus': pq.get('corpus'),
            'num_candidates': len(pq['candidates']),
            'relevant_doc_ids': sorted(relevant),
        })

    n_j = len(judgments)
    g0 = sum(1 for j in judgments if j['a1'] == 0)
    g1 = sum(1 for j in judgments if j['a1'] == 1)
    g2 = sum(1 for j in judgments if j['a1'] == 2)

    qrels = {
        '_schema_note': 'SINGLE-RATER human qrels (v2 blind pool, annotator 1 only). '
                        'relevant_doc_ids = binarized at grade >= 1. NO inter-annotator '
                        'reliability estimate exists for this set.',
        '_WARNING': 'SINGLE RATER. No cohen_kappa is reported because there is no '
                    'second annotator. Treat as unvalidated single-rater judgments. '
                    'Do not cite as agreement or as a reliability estimate.',
        'resolution_method': 'SINGLE RATER (annotator 1), binarized at >=1',
        'n_raters': 1,
        'cohen_kappa': None,
        'percent_agreement': None,
        'n_queries': len(queries),
        'n_judgments': n_j,
        'grade_distribution': {'0': g0, '1': g1, '2': g2},
        'queries': queries,
    }

    audit = {
        '_note': 'Per-judgment single-rater labels. a2 is null by construction; no '
                 'agreement statistics are computed from this file.',
        'n_raters': 1,
        'cohen_kappa': None,
        'percent_agreement': None,
        'per_query_stats': per_query,
        'judgments': judgments,
    }

    json.dump(qrels, open(OUT_QRELS, 'w'), indent=2)
    json.dump(audit, open(OUT_AUDIT, 'w'), indent=2)

    print("=" * 68)
    print("SINGLE-RATER QRELS BUILT")
    print("=" * 68)
    print(f"  queries            : {len(queries)}")
    print(f"  judgments          : {n_j}")
    print(f"  grades             : 0={g0}  1={g1}  2={g2}")
    print(f"  cohen_kappa        : None  (single rater - NOT computable)")
    print(f"  percent_agreement  : None  (single rater - NOT computable)")
    print()
    print(f"  -> {os.path.relpath(OUT_QRELS, BASE)}")
    print(f"  -> {os.path.relpath(OUT_AUDIT, BASE)}")
    print()
    print("  Run the evaluation with:")
    print("    HUMAN_QRELS=data/qrels_human_annotated_v2_singlerater.json \\")
    print("    HUMAN_AUDIT=data/qrels_human_annotated_v2_singlerater_audit.json \\")
    print("    python evaluation/paired_bootstrap_graded.py")
    print("=" * 68)


if __name__ == '__main__':
    main()
