"""Is the deeper reference actually worse ground truth?

The depth sweep treats K>5 as a sensitivity probe, not a correction, on the
ground that larger K admits progressively worse term-overlap documents. This
script measures that directly: for every K, the distribution of the OVERLAP score
of the documents actually admitted into the reference.

If the "lexical noise" claim is right, the overlap distribution must degrade
with K -- and the mass admitted beyond K=5 should sit near the bottom of the
ranking (overlap 1, i.e. a single shared query term).

This is what distinguishes a sensitivity bound from a fix: a K whose reference is
dominated by overlap-1 documents is a looser instrument, not a better one.
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'evaluation'))

from temporal.temporal_parser import TemporalParser

STOP = {"and", "for", "the", "in", "to", "of", "with", "from", "on", "at", "by",
        "this", "that", "it", "as", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "will", "would", "could", "should", "not", "no",
        "since", "before", "after", "during"}
parser = TemporalParser()
DEPTHS = [5, 10, 20, 50]


def get_terms(t):
    return {w for w in re.findall(r'\b[a-z]{3,}\b', t.lower()) if w not in STOP}


def main():
    chunks = [json.loads(l) for l in open(os.path.join(REPO, 'data', 'chunks.jsonl'),
                                         encoding='utf-8') if l.strip()]
    cur = json.load(open(os.path.join(REPO, 'data', 'qrels_expanded.json'),
                         encoding='utf-8'))
    scored = [(q['query_id'], q['query_text']) for q in cur if q.get('relevant_doc_ids')]

    chunk_toks = [(c.get('publication_year'), c['parent_doc_id'], get_terms(c['text']))
                  for c in chunks]

    # per-query ordered (pid, overlap) after date filter + threshold
    pools = []
    for qid, text in scored:
        ti = parser.parse(text)
        sy, ey = ti.get('start_year'), ti.get('end_year')
        qt = get_terms(text)
        scored_chunks = []
        for doc_year, pid, cterms in chunk_toks:
            if sy is not None and ey is not None:
                if doc_year is None or not (sy <= doc_year <= ey):
                    continue
            ov = len(qt & cterms)
            if ov > 0:
                scored_chunks.append((pid, ov))
        scored_chunks.sort(key=lambda x: -x[1])
        seen, dedup = set(), []
        for pid, ov in scored_chunks:
            if pid not in seen:
                seen.add(pid)
                dedup.append((pid, ov))
        if dedup:
            thr = max(1, dedup[0][1] - 1)
            dedup = [d for d in dedup if d[1] >= thr]
        pools.append(dedup)

    print(f"scored queries: {len(pools)}")
    print(f"\n{'K':>4} {'mean refs':>11} {'mean overlap':>13} {'median':>8} "
          f"{'% overlap<=1':>13} {'% overlap<=2':>13} {'min':>5}")
    out = {}
    for K in DEPTHS:
        ovs = []
        for pool in pools:
            ovs.extend(o for _, o in pool[:K])
        ovs = np.array(ovs)
        frac1 = float((ovs <= 1).mean())
        frac2 = float((ovs <= 2).mean())
        out[K] = {'n_refs': int(len(ovs)), 'mean_refs_per_query': round(len(ovs) / len(pools), 2),
                  'mean_overlap': round(float(ovs.mean()), 3),
                  'median_overlap': float(np.median(ovs)),
                  'min_overlap': int(ovs.min()),
                  'frac_overlap_le_1': round(frac1, 4),
                  'frac_overlap_le_2': round(frac2, 4)}
        print(f"{K:>4} {len(ovs)/len(pools):>11.2f} {ovs.mean():>13.3f} "
              f"{np.median(ovs):>8.1f} {frac1*100:>12.1f}% {frac2*100:>12.1f}% {ovs.min():>5}")

    # how much of each depth's reference is NEW relative to K=5
    print(f"\n{'K':>4} {'refs beyond K=5':>18} {'mean overlap of that tail':>26}")
    for K in DEPTHS:
        tail = []
        for pool in pools:
            tail.extend(o for _, o in pool[5:K])
        if tail:
            t = np.array(tail)
            print(f"{K:>4} {len(t):>18} {t.mean():>26.3f}   "
                  f"({(t<=1).mean()*100:.1f}% at overlap<=1)")
        else:
            print(f"{K:>4} {0:>18}")

    with open(os.path.join(REPO, 'evaluation', 'qrels_depth_noise.json'),
              'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nwrote evaluation/qrels_depth_noise.json")


if __name__ == '__main__':
    main()
