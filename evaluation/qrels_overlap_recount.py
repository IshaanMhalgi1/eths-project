"""
Exact recount of the qrels-overlap figure, with every intermediate count shown.

The v1 report quoted "micro-Jaccard 0.08" without showing the raw counts, which
made it impossible to tell whether it reflected a small numerator or a large
denominator. This spells out |H|, |T|, |H n T|, and |H u T| per query and in
total, using the corpus-matched term-overlap qrels (the same construction the
ablations used) rather than a union of the small and expanded files.
"""
import json
import os

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')

human = json.load(open(os.path.join(DATA, 'qrels_human_annotated.json')))
small = json.load(open(os.path.join(DATA, 'qrels_small.json')))
expanded = json.load(open(os.path.join(DATA, 'qrels_expanded.json')))

# Corpus-matched: a small-corpus query is judged against qrels_small, otherwise
# against qrels_expanded. This mirrors how the ablations were actually run.
small_by_id = {q['query_id']: q for q in small}
exp_by_id = {q['query_id']: q for q in expanded}


def term_relevant(q):
    src = small_by_id if q['corpus'] == 'small' else exp_by_id
    entry = src.get(q['query_id'])
    return set(entry['relevant_doc_ids']) if entry else set()


print("=" * 92)
print("RELEVANT-SET OVERLAP: term-overlap vs human, with raw counts")
print("=" * 92)
print(f"{'query':<7}{'|H|':<6}{'|T|':<6}{'|HnT|':<7}{'|HuT|':<7}{'Jaccard':<10}{'query text'}")
sumH = sumT = sumI = 0
for q in human['queries']:
    H = set(q['relevant_doc_ids'])
    T = term_relevant(q)
    I = H & T
    U = H | T
    sumH += len(H)
    sumT += len(T)
    sumI += len(I)
    j = len(I) / len(U) if U else 0.0
    print(f"{q['query_id']:<7}{len(H):<6}{len(T):<6}{len(I):<7}{len(U):<7}{j:<10.3f}{q['query_text'][:30]}")

sumU = sumH + sumT - sumI
zero_overlap = sum(1 for q in human['queries']
                   if not (set(q['relevant_doc_ids']) & term_relevant(q)))
max_overlap = max(len(set(q['relevant_doc_ids']) & term_relevant(q))
                  for q in human['queries'])
max_overlap_qid = max(human['queries'],
                      key=lambda q: len(set(q['relevant_doc_ids']) & term_relevant(q)))['query_id']

print(f"\n{'-'*92}")
print(f"{'TOTAL':<7}{sumH:<6}{sumT:<6}{sumI:<7}{sumU:<7}{sumI/sumU:<10.3f}")

print(f"""
MICRO-JACCARD = |H n T| / |H u T| = {sumI} / {sumU} = {sumI/sumU:.3f}

Reading the raw counts:
  - The human qrels mark {sumH} documents relevant across 10 queries.
  - The term-overlap qrels mark {sumT} relevant for the SAME 10 queries
    (always exactly 5 per query, by construction of the term-overlap builder).
  - They share {sumI} documents.
  - The union is {sumU}, so the {sumI/sumU:.3f} figure is a SMALL NUMERATOR
    ({sumI} shared documents out of {sumH + sumT} judgments), not an artifact
    of a large denominator.

Per-query, {zero_overlap} of 10 queries have ZERO overlap between the two
relevant sets. The maximum per-query overlap is {max_overlap} document(s)
({max_overlap_qid}). The human qrels are also markedly broader: they mark
{sumH} relevant against the term-overlap qrels' {sumT}, i.e. {sumH/sumT:.1f}x
as many, which is expected since annotators could mark partial relevance and
the pool was deeper than 5.

CAVEAT ON WHAT THIS DOES AND DOES NOT SHOW.
Low overlap is necessary-but-not-sufficient evidence that the term-overlap
qrels are wrong. Two annotators with kappa=0.45 would also produce a
low-overlap set. The overlap statistic establishes that the two qrels
measure different things; it does not by itself establish which one is
right. The pilot's own reliability problem (weak inter-annotator
agreement) is the reason this cannot currently be settled.
""")
