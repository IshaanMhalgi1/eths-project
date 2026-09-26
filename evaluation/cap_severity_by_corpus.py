"""
Does the top-5 qrels cap bite EQUALLY across the 10-year and 100-year corpora?

This matters because the corpus-span temporal comparison in
TEMPORAL_RERANKING_FINDING_REPORT.md contrasts the same rankers across two
corpora. If the cap is more binding in one corpus, the contrast is confounded
by differential truncation severity rather than reflecting corpus behaviour.

Measures, per corpus:
  1. Cap check            - is the reference hard-capped, and at what value?
  2. Retrievability       - of the N reference docs, how many does each retriever
                            surface in its top-20? A low number means the
                            reference is describing documents the retriever
                            struggles to find, i.e. the cap is describing a small
                            and atypical slice of the corpus.
  3. Ceiling headroom     - how many of the retriever's top-20 are NOT in the
                            5-doc reference. High headroom = most of what the
                            retriever surfaces cannot be credited.
  4. Rank5->rank6 sharpness - if the term-overlap generator's 5th and 6th ranked
                            documents are nearly tied, the cut is arbitrary and
                            discards near-equally-relevant documents.
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from retrieval.bm25 import search_bm25, DEFAULT_INDEX, SMALL_INDEX
from retrieval.dense import search_dense

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
TOPK = 20

CORPORA = [
    ('10-year (1800-1810)', 'qrels_small.json', SMALL_INDEX),
    ('100-year (1800-1900)', 'qrels_expanded.json', DEFAULT_INDEX),
]


def load(name):
    d = json.load(open(os.path.join(DATA, name)))
    return d['queries'] if isinstance(d, dict) else d


def main():
    from collections import Counter
    print("=" * 78)
    print("DOES THE TOP-5 CAP BITE EQUALLY ACROSS CORPORA?")
    print("=" * 78)

    summary = {}
    for label, qfile, index in CORPORA:
        qs = load(qfile)
        kept = [q for q in qs if q.get('relevant_doc_ids')]
        caps = Counter(len(q['relevant_doc_ids']) for q in kept)

        print()
        print("-" * 78)
        print(f"{label}   ({len(kept)} evaluated queries, {qfile})")
        print("-" * 78)
        print(f"  cap distribution      : {dict(sorted(caps.items()))}")
        capped_at_5 = caps.get(5, 0)
        print(f"  queries capped at 5   : {capped_at_5}/{len(kept)} "
              f"({100*capped_at_5/len(kept):.0f}%)")

        # retrievability + headroom
        bm25_hits, dense_hits = [], []
        bm25_head, dense_head = [], []
        ref_sizes = []
        for q in kept:
            rel = set(q['relevant_doc_ids'])
            ref_sizes.append(len(rel))
            text = q['query_text']
            for fn, hits, head in ((search_bm25, bm25_hits, bm25_head),
                                   (search_dense, dense_hits, dense_head)):
                try:
                    res = fn(text, size=TOPK, index_name=index)
                except TypeError:
                    res = fn(text, size=TOPK)
                got = [r['parent_doc_id'] for r in res][:TOPK]
                hits.append(sum(1 for d in got if d in rel))
                head.append(sum(1 for d in got if d not in rel))

        n_ref = sum(ref_sizes)
        for name, hits, head in (('BM25', bm25_hits, bm25_head),
                                 ('Dense', dense_hits, dense_head)):
            mean_hit = sum(hits) / len(hits)
            mean_head = sum(head) / len(head)
            reach = mean_hit / (n_ref / len(ref_sizes))
            print(f"  {name:<6} reference docs in top-{TOPK} : "
                  f"{mean_hit:.2f}/{n_ref/len(ref_sizes):.2f} "
                  f"({100*reach:.0f}% of the reference reachable)")
            print(f"  {name:<6} uncredited docs in top-{TOPK}: {mean_head:.2f}")

        summary[label] = {
            'n_queries': len(kept),
            'caps': dict(sorted(caps.items())),
            'bm25_reach': sum(bm25_hits) / len(bm25_hits),
            'dense_reach': sum(dense_hits) / len(dense_hits),
            'bm25_headroom': sum(bm25_head) / len(bm25_head),
            'dense_headroom': sum(dense_head) / len(dense_head),
        }

    print()
    print("=" * 78)
    print("COMPARISON")
    print("=" * 78)
    a, b = CORPORA[0][0], CORPORA[1][0]
    print(f"{'measure':<34}{a:<22}{b}")
    print("-" * 78)
    for key, lab in (('bm25_reach', 'BM25 ref-docs reachable/top20'),
                     ('dense_reach', 'Dense ref-docs reachable/top20'),
                     ('bm25_headroom', 'BM25 uncredited docs/top20'),
                     ('dense_headroom', 'Dense uncredited docs/top20')):
        print(f"{lab:<34}{summary[a][key]:<22.2f}{summary[b][key]:.2f}")
    print()
    print("Caps:")
    print(f"  {a}: {summary[a]['caps']}")
    print(f"  {b}: {summary[b]['caps']}")


if __name__ == '__main__':
    main()
