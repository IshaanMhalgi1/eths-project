"""
Characterise annotator 1's blind v2 labels against retriever rank.

Uses the post-hoc scored sidecar (never shown to annotators) to ask: did the
retrievers' higher-ranked candidates get judged relevant more often than
their lower-ranked ones? A monotone trend is evidence the pool carries real
signal; a flat or inverted trend would mean the pool is largely noise and
the whole exercise is measuring annotation of nothing.
"""
import json
from collections import defaultdict

pool = json.load(open('data/annotation/annotation_pools.json'))
a1 = json.load(open('data/annotation_v2_annotator1.json'))
scored = json.load(open('data/annotation_pools_with_scores.json'))

smap = {q['query_id']: q for q in scored['queries']}
labels = {}
for q in a1['queries']:
    for c in q['candidates']:
        labels[(q['query_id'], c['canonical_parent_doc_id'])] = c['relevance_annotator1']


def rel(g):
    return 1 if g is not None and g >= 1 else 0


# The sidecar stores per-candidate raw scores, not ranks. Ranking WITHIN each
# retriever and each query is valid (raw scales are not comparable ACROSS
# retrievers, which is exactly the bug the v1 pool had).
per_ret = defaultdict(lambda: defaultdict(list))
skipped = 0

for q in pool['queries']:
    qid = q['query_id']
    by_doc = {c['canonical_parent_doc_id']: c.get('retriever_scores', {})
              for c in smap.get(qid, {}).get('candidates', [])}

    for rname in ('bm25', 'dense', 'hybrid'):
        # rank this retriever's docs for this query
        scored_pairs = [
            (doc, rs.get(rname)) for doc, rs in by_doc.items()
            if rs.get(rname) is not None
        ]
        scored_pairs.sort(key=lambda t: -t[1])
        for rank, (doc, _s) in enumerate(scored_pairs, start=1):
            if (qid, doc) not in labels:
                skipped += 1
                continue
            y = rel(labels[(qid, doc)])
            per_ret[rname]['all'].append(y)
            b = '1-5' if rank <= 5 else ('6-10' if rank <= 10 else '11-20')
            per_ret[rname][b].append(y)

print("Rank derived by sorting each retriever's own scores within each query.")
if skipped:
    print(f"(skipped {skipped} scored entries with no annotator-1 label)")
print("-" * 72)

for rname in ('bm25', 'dense', 'hybrid'):
    d = per_ret.get(rname, {})
    cells = []
    for b in ('1-5', '6-10', '11-20', 'all'):
        v = d.get(b, [])
        cells.append(f"{100*sum(v)/len(v):.0f}%(n={len(v)})" if v else "n/a")
    print(f"{rname:<8}{cells[0]:>14}{cells[1]:>14}{cells[2]:>14}{cells[3]:>14}")

# ---------- per-query richness ----------
print()
print("=" * 72)
print("PER-QUERY JUDGMENT PROFILE (annotator 1)")
print("=" * 72)
print(f"{'query':<7}{'cands':>6}{'g0':>5}{'g1':>5}{'g2':>5}{'rel%':>7}  query text")
print("-" * 72)
for q in pool['queries']:
    qid = q['query_id']
    gs = [labels[(qid, c['canonical_parent_doc_id'])] for c in q['candidates']]
    g0, g1, g2 = gs.count(0), gs.count(1), gs.count(2)
    print(f"{qid:<7}{len(gs):>6}{g0:>5}{g1:>5}{g2:>5}{100*(g1+g2)/len(gs):>6.0f}%  {q['query_text'][:44]}")

# ---------- is any query degenerate? ----------
print()
allrel = [len(q['candidates']) for q in pool['queries']]
print(f"pool sizes: min={min(allrel)} max={max(allrel)} total={sum(allrel)}")
thin = [(q['query_id'], len(q['candidates'])) for q in pool['queries'] if len(q['candidates']) < 10]
print("queries with <10 candidates (too thin to support a metric):", thin)
