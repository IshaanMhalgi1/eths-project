"""Per-cluster and per-query breakdown of the class-level headline contrasts.

Two questions this answers:

1. WHAT ARE THE 8 EXPANDED CLUSTERS? The corpus spans 7 decades (1800-1869), so
   7 decade buckets are expected. The 8th is a `no_year` catch-all: queries
   whose text contains no 4-digit year in 1800-1869. That bucket is
   heterogeneous -- it mixes genuinely non-temporal queries with era-named ones
   ("Jeffersonian Era") whose temporal intent the parser still resolves from a
   period table rather than from a literal year. So "no literal year" is NOT the
   same as "no temporal intent", and this script reports both.

2. HOW DOES EACH HEADLINE CONTRAST BEHAVE PER CLUSTER AND PER QUERY? For every
   cluster: the mean paired Recall@10 difference. For every headline contrast:
   how many of the 88 queries the winner improved, left unchanged, or lost.

Reads stored per-query metrics; no ranker, corpus, or index access.
"""
import json
import os
import re
import statistics as st
import sys
from collections import OrderedDict

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from temporal.temporal_parser import TemporalParser

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'evaluation', 'cluster_breakdown.json')
parser = TemporalParser()

# class representatives from equivalence_classes.json (expanded)
REPS = OrderedDict([
    ('C0', 'BM25'),          # {BM25, Hybrid, Hybrid+Metadata, MetadataOnly}
    ('C1', 'Dense'),
    ('C2', 'Hybrid+Temporal'),   # {BM25+Temporal, Hybrid+Temporal}
    ('C3', 'Final'),
    ('C4', 'Dense+Temporal'),
    ('C5', 'TemporalOnly'),
])

HEADLINE = [
    ('C2', 'C0', 'temporal(beta=0.5) vs BM25-family'),
    ('C5', 'C0', 'TemporalOnly vs BM25-family'),
    ('C3', 'C0', 'Final vs BM25-family'),
    ('C0', 'C1', 'BM25-family vs Dense'),
    ('C2', 'C1', 'temporal vs Dense'),
    ('C2', 'C4', 'temporal vs Dense+Temporal'),
    ('C4', 'C1', 'Dense+Temporal vs Dense'),
]


def decade_of(text):
    ys = [int(y) for y in re.findall(r"\b(1[0-8][0-9]{2})\b", text) if 1800 <= int(y) <= 1869]
    if not ys:
        return "no_year", []
    return f"{(int(st.median(ys)) // 10) * 10}s", ys


def main():
    with open(os.path.join(REPO, 'evaluation', 'ablation_matrix_results.json'),
              encoding='utf-8') as f:
        abl = json.load(f)
    with open(os.path.join(REPO, 'data', 'qrels_expanded.json'), encoding='utf-8') as f:
        queries = [q for q in json.load(f) if q.get('relevant_doc_ids')]

    m = abl['metrics']['expanded']
    n = len(queries)
    series = {c: np.array(m[r]['per_query']['r'], dtype=float) for c, r in REPS.items()}

    rows = []
    for q in queries:
        dec, literal = decade_of(q['query_text'])
        ti = parser.parse(q['query_text'])
        rows.append({
            'query_id': q['query_id'],
            'temporal_type': q.get('temporal_type'),
            'text': q['query_text'],
            'cluster': dec,
            'literal_years': literal,
            'parser_start': ti.get('start_year'),
            'parser_end': ti.get('end_year'),
            'has_temporal_intent': ti.get('start_year') is not None,
            'n_refs': len(q['relevant_doc_ids']),
        })

    # ---------------- cluster inventory ----------------
    order = [f"{d}s" for d in range(1800, 1870, 10)] + ['no_year']
    clusters = OrderedDict()
    for c in order:
        members = [r for r in rows if r['cluster'] == c]
        if not members:
            continue
        with_intent = sum(1 for r in members if r['has_temporal_intent'])
        clusters[c] = {
            'n_queries': len(members),
            'query_ids': [r['query_id'] for r in members],
            'temporal_types': {t: sum(1 for r in members if r['temporal_type'] == t)
                               for t in sorted({r['temporal_type'] for r in members})},
            'n_with_parser_temporal_intent': with_intent,
            'n_without_temporal_intent': len(members) - with_intent,
            'mean_refs_per_query': round(float(np.mean([r['n_refs'] for r in members])), 2),
        }

    print("=" * 100)
    print(f"EXPANDED CLUSTERS  (corpus spans 7 decades 1800-1869; n={n} queries)")
    print("=" * 100)
    print(f"{'cluster':<10}{'n':>4}  {'parser intent':>14}  {'types':<34}{'refs/q':>7}")
    for c, info in clusters.items():
        types = ",".join(f"{k}:{v}" for k, v in info['temporal_types'].items())
        print(f"{c:<10}{info['n_queries']:>4}  "
              f"{info['n_with_parser_temporal_intent']:>6}/{info['n_queries']:<7} "
              f"{types:<34}{info['mean_refs_per_query']:>7.2f}")
    print(f"{'TOTAL':<10}{sum(i['n_queries'] for i in clusters.values()):>4}")

    # ---------------- per-cluster effect table ----------------
    cids = list(REPS)
    print("\n" + "=" * 100)
    print("PER-CLUSTER PAIRED Recall@10 DIFFERENCE (winner - loser)")
    print("=" * 100)
    header = f"{'cluster':<10}{'n':>4}"
    for a, b, _ in HEADLINE:
        header += f"{REPS[a][:11]+' > '+REPS[b][:11]:>26}"
    print(header)
    per_cluster = {}
    for c in order:
        if c not in clusters:
            continue
        idx = [i for i, r in enumerate(rows) if r['cluster'] == c]
        line = f"{c:<10}{len(idx):>4}"
        rec = {'n': len(idx)}
        for a, b, lbl in HEADLINE:
            d = float(np.mean(series[a][idx] - series[b][idx]))
            rec[lbl] = d
            line += f"{d:>+26.4f}"
        per_cluster[c] = rec
        print(line)
    line = f"{'ALL':<10}{n:>4}"
    for a, b, lbl in HEADLINE:
        line += f"{float(np.mean(series[a]-series[b])):>+26.4f}"
    print(line)

    # ---------------- per-query win/loss/tie ----------------
    print("\n" + "=" * 100)
    print("PER-QUERY BREAKDOWN of the 88 scored queries (Recall@10, winner's perspective)")
    print("=" * 100)
    print(f"{'contrast':<40}{'improved':>10}{'unchanged':>11}{'worse':>8}"
          f"{'mean d':>10}{'d in no_year':>14}")
    breakdown = {}
    for a, b, lbl in HEADLINE:
        d = series[a] - series[b]
        imp = int((d > 0).sum())
        tie = int((d == 0).sum())
        wor = int((d < 0).sum())
        ny = [i for i, r in enumerate(rows) if r['cluster'] == 'no_year']
        d_ny = float(np.mean(d[ny]))
        breakdown[lbl] = {'improved': imp, 'unchanged': tie, 'worse': wor,
                          'mean_diff': float(d.mean()), 'mean_diff_no_year': d_ny}
        print(f"{lbl:<40}{imp:>10}{tie:>11}{wor:>8}{d.mean():>+10.4f}{d_ny:>+14.4f}")

    # which clusters drive C2>C0
    print("\n  C2 > C0 detail: effect by cluster, and year-bearing vs not")
    for c in order:
        if c not in clusters:
            continue
        idx = [i for i, r in enumerate(rows) if r['cluster'] == c]
        d = series['C2'][idx] - series['C0'][idx]
        print(f"    {c:<10} n={len(idx):<3} mean d={d.mean():+.4f}  "
              f"improved={int((d>0).sum()):<3} unchanged={int((d==0).sum()):<3} "
              f"worse={int((d<0).sum()):<3}")

    out = {'n_queries': n, 'cluster_definition': {
        'rule': 'median of all 4-digit years 1800-1869 in the query text, '
                'floored to decade; "no_year" if the text contains none',
        'note': 'no_year is a catch-all mixing non-temporal queries with '
                'era-named queries whose intent the parser resolves without a '
                'literal year'},
        'clusters': clusters, 'per_cluster_effects': per_cluster,
        'per_query_breakdown': breakdown, 'rows': rows}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote evaluation/{os.path.basename(OUT)}")


if __name__ == '__main__':
    main()
