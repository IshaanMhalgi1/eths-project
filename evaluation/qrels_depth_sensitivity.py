"""Qrels-depth (K) sensitivity for the class-level contrasts.

The term-overlap reference is a hard top-K cut of an overlap ranking, so every
number in the project is conditioned on K=5. This re-generates the reference at
K in {5, 10, 20, 50} and re-runs the class-level contrasts at each depth.

WHAT IS PARAMETERISED: only the hard cap `[:K]` that truncates the overlap-ranked
candidate list. The *similarity* filter just above it -- threshold =
max(1, top_overlap - 1) -- is NOT a depth parameter and is held fixed, so the
candidate pool is identical across K and only the truncation point moves.

K=5 MUST REPRODUCE THE CURRENT FILE EXACTLY. That is asserted, not assumed. This
holds for data/qrels_expanded.json (verified: 0/108 mismatches) and the sweep runs
there. It does NOT hold for data/qrels_small.json, which is excluded: its
references are consecutive chunk ids of the same kind (`train_257..train_261`,
`train_141..train_145`) that match neither generate_expanded_qrels.py's
overlap ranking nor generate_qrels.py's self-referential construction -- 19/19
mismatches under both, and 19/19 still after restricting the pool to the small
index's own 3,543 documents. The small reference's generator is not present in
the repository, so no depth sweep over it would be a controlled comparison. It is
reported as excluded rather than approximated.

Retrieval does not depend on K, so each ranker's deduplicated top-10 is computed
ONCE and scored against every K's reference. That also guarantees all depths see
byte-identical system output.

WHY THIS BOUNDS BUT DOES NOT FIX THE TRUNCATION EFFECT: enlarging K admits
lower-overlap documents into the reference. Those are progressively worse
term-overlap matches, so the reference becomes lexically noisier and less like a
relevance judgment as K grows. A stable effect across K therefore shows the
conclusion is not an artifact of the K=5 cut -- but the K>5 references are
arguably WORSE ground truth, not better, so this is a sensitivity bound and not
a correction.

Writes evaluation/qrels_depth_sensitivity.json. Analysis only: no ranker, corpus,
or stored-reference file is modified.
"""
import json
import os
import re
import sys
from collections import Counter, OrderedDict
from math import comb

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from temporal.temporal_parser import TemporalParser
from evaluation.ablation_rankers import get_ranker
from ranking.shared_hybrid import SHARED_FETCH_SIZE
from class_level_significance import (paired_bootstrap, cluster_bootstrap,
                                      sign_test_p, decade_of, N_BOOT, SEED)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS = os.path.join(REPO, 'data', 'chunks.jsonl')
OUT = os.path.join(REPO, 'evaluation', 'qrels_depth_sensitivity.json')

DEPTHS = [5, 10, 20, 50]
FETCH_MULT, EVAL_K = 5, 10

EXPANDED_CLASSES = OrderedDict([
    ('C0', 'BM25'), ('C1', 'Dense'), ('C2', 'Hybrid+Temporal'),
    ('C3', 'Final'), ('C4', 'Dense+Temporal'), ('C5', 'TemporalOnly')])
SMALL_CLASSES = OrderedDict([
    ('C0', 'BM25'), ('C1', 'Dense'), ('C2', 'Hybrid'),
    ('C3', 'Hybrid+Temporal'), ('C4', 'Final'),
    ('C5', 'BM25+Temporal'), ('C6', 'Dense+Temporal'), ('C7', 'TemporalOnly')])

HEADLINE = {
    'expanded': [('C2', 'C0', 'temporal(beta=.5) vs BM25-family'),
                 ('C5', 'C0', 'TemporalOnly vs BM25-family'),
                 ('C0', 'C1', 'BM25-family vs Dense')],
    'small': [('C3', 'C0', 'temporal(beta=.3) vs BM25-family'),
              ('C5', 'C0', 'BM25+Temporal vs BM25-family'),
              ('C0', 'C1', 'BM25-family vs Dense')],
}

STOP = {"and", "for", "the", "in", "to", "of", "with", "from", "on", "at", "by",
        "this", "that", "it", "as", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "will", "would", "could", "should", "not", "no",
        "since", "before", "after", "during"}
parser = TemporalParser()


def get_terms(text):
    return {w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in STOP}


def build_candidates(queries, chunks):
    """Threshold-filtered, deduplicated, overlap-ranked candidate list per query.

    This is the part of the generator that does NOT depend on the reference
    depth: date filtering, term-overlap scoring, the max(1, top_overlap-1)
    similarity threshold, and parent_doc_id dedup. Computed ONCE and sliced to
    each K below, so every depth is guaranteed to see an identical candidate
    pool and only the truncation point moves.
    """
    chunk_toks = [(c.get('publication_year'), c['parent_doc_id'],
                   get_terms(c['text'])) for c in chunks]
    out = []
    for q in queries:
        text = q['query_text']
        ti = parser.parse(text)
        sy, ey = ti.get('start_year'), ti.get('end_year')
        qt = get_terms(text)
        scored = []
        for doc_year, pid, cterms in chunk_toks:
            if sy is not None and ey is not None:
                if doc_year is None or not (sy <= doc_year <= ey):
                    continue
            ov = len(qt & cterms)
            if ov > 0:
                scored.append({'parent_doc_id': pid, 'overlap': ov})
        scored.sort(key=lambda x: x['overlap'], reverse=True)
        seen, dedup = set(), []
        for sc in scored:
            if sc['parent_doc_id'] not in seen:
                seen.add(sc['parent_doc_id'])
                dedup.append(sc)
        if not dedup:
            out.append([])
        else:
            thr = max(1, dedup[0]['overlap'] - 1)
            out.append([d['parent_doc_id'] for d in dedup if d['overlap'] >= thr])
    return out


def build_qrels(queries, candidates, depth):
    """Apply the reference depth K to a precomputed candidate list."""
    out = []
    for q, cands in zip(queries, candidates):
        r = dict(q)
        r['relevant_doc_ids'] = cands[:depth]
        out.append(r)
    return out


def top10_lists(corpus, classes, queries):
    """Deduplicated top-10 parent_doc_id list per (class, query). Retrieval is
    independent of the reference depth, so this is computed once."""
    res = {}
    for cid, rep in classes.items():
        fn = get_ranker(rep, corpus=corpus)
        per_q = []
        for q in queries:
            rows = fn(q['query_text'], size=EVAL_K * FETCH_MULT)
            seen, dedup = set(), []
            for r in rows:
                p = r['parent_doc_id']
                if p not in seen:
                    seen.add(p)
                    dedup.append(p)
            per_q.append(dedup[:EVAL_K])
        res[cid] = per_q
    return res


def score(top10, refs):
    r_list, m_list, h_list = [], [], []
    for docs, true in zip(top10, refs):
        true = set(true)
        hits = sum(1 for d in docs if d in true)
        r_list.append(hits / max(len(true), 1) if true else 0.0)
        mrr = 0.0
        for i, d in enumerate(docs):
            if d in true:
                mrr = 1.0 / (i + 1)
                break
        m_list.append(mrr)
        h_list.append(hits)
    return r_list, m_list, h_list


def main():
    with open(CHUNKS, encoding='utf-8') as f:
        chunks = [json.loads(l) for l in f if l.strip()]
    print(f"loaded {len(chunks)} chunks")

    datasets = {}
    for corpus, path in (('expanded', 'data/qrels_expanded.json'),):
        with open(os.path.join(REPO, path), encoding='utf-8') as f:
            current = json.load(f)
        queries = [{k: v for k, v in q.items() if k != 'relevant_doc_ids'}
                   for q in current]
        scored_mask = [bool(q.get('relevant_doc_ids')) for q in current]
        datasets[corpus] = {'queries': queries, 'current': current,
                            'scored_mask': scored_mask, 'classes': EXPANDED_CLASSES}
        print(f"{corpus}: {len(queries)} queries, {sum(scored_mask)} scored")
        d = datasets[corpus]
        print(f"  building candidate pools for {corpus} ...")
        d['candidates'] = build_candidates(queries, chunks)
        lens = [len(c) for c in d['candidates']]
        print(f"  candidate pool sizes: min={min(lens)} max={max(lens)} "
              f"mean={np.mean(lens):.1f}")

    # Exclude the small corpus: its reference is not reproducible at K=5 by any
    # generator in the repo, so a depth sweep over it would not be controlled.
    excluded = {}
    with open(os.path.join(REPO, 'data', 'qrels_small.json'), encoding='utf-8') as f:
        small_cur = json.load(f)
    small_q = [{k: v for k, v in q.items() if k != 'relevant_doc_ids'} for q in small_cur]
    small_c = build_candidates(small_q, chunks)
    small_g = build_qrels(small_q, small_c, 5)
    n_bad = sum(1 for a, b in zip(small_g, small_cur)
                if a['relevant_doc_ids'] != b['relevant_doc_ids'])
    excluded['small'] = {
        'n_queries': len(small_cur),
        'k5_mismatches': n_bad,
        'reason': 'reference not reproducible at K=5 by generate_expanded_qrels.py '
                  'logic; consecutive train_* chunk ids suggest a different, '
                  'absent generator. Excluded rather than approximated.'}
    print(f"\nsmall corpus EXCLUDED: {n_bad}/{len(small_cur)} K=5 mismatches")

    # ---------------- generate + verify K=5 reproduces current ----------------
    print("\n=== K=5 reproduction check (must be EXACT) ===")
    for corpus, d in datasets.items():
        gen5 = build_qrels(d['queries'], d['candidates'], 5)
        ok = True
        for a, b in zip(gen5, d['current']):
            if a['relevant_doc_ids'] != b['relevant_doc_ids']:
                ok = False
                break
        dist_g = Counter(len(q['relevant_doc_ids']) for q in gen5)
        dist_c = Counter(len(q['relevant_doc_ids']) for q in d['current'])
        print(f"  {corpus:<9} relevant_doc_ids identical: {ok}   "
              f"gen dist={dict(sorted(dist_g.items()))}  current dist={dict(sorted(dist_c.items()))}")
        if not ok:
            raise SystemExit(f"FAIL: K=5 does not reproduce {corpus} reference")
        d['qrels_by_k'] = {}

    # ---------------- retrieval cache ----------------
    for corpus, d in datasets.items():
        idx = [i for i, m in enumerate(d['scored_mask']) if m]
        qs = [d['queries'][i] for i in idx]
        print(f"\nretrieving {corpus} top-10 for {len(d['classes'])} classes "
              f"over {len(qs)} scored queries ...")
        d['top10'] = top10_lists(corpus, d['classes'], qs)
        d['clusters'] = np.array([hash(decade_of(q['query_text'])) for q in qs])
        labels = sorted({decade_of(q['query_text']) for q in qs})
        d['cluster_labels'] = labels
        d['cluster_ids'] = np.array([labels.index(decade_of(q['query_text'])) for q in qs])
        d['n_clusters'] = len(labels)
        print(f"   clusters: {len(labels)} -> {Counter(decade_of(q['query_text']) for q in qs)}")

    # ---------------- sweep K ----------------
    report = {'depths': DEPTHS, 'excluded_corpora': excluded, 'corpora': {}}
    for corpus, d in datasets.items():
        n = len(d['top10']['C0'])
        k = len(d['classes'])
        n_pairs = k * (k - 1) // 2
        alpha = 0.05 / n_pairs
        rng = np.random.default_rng(SEED)
        idxq = rng.integers(0, n, size=(N_BOOT, n))
        print(f"\n{'='*100}\n{corpus.upper()}  n={n}  classes={k}  "
              f"class pairs={n_pairs}  Bonferroni alpha={alpha:.6f}  "
              f"clusters={d['n_clusters']}\n{'='*100}")

        rows = []
        for depth in DEPTHS:
            gen = build_qrels(d['queries'], d['candidates'], depth)
            refs = [gen[i]['relevant_doc_ids'] for i in range(len(gen)) if d['scored_mask'][i]]
            nref = [len(r) for r in refs]
            scored = {}
            for cid in d['classes']:
                r_list, m_list, h_list = score(d['top10'][cid], refs)
                scored[cid] = {'r': np.array(r_list), 'mrr': np.array(m_list),
                               'hits': np.array(h_list)}
            mean_refs = float(np.mean(nref))
            rec = {'depth': depth, 'mean_refs_per_query': round(mean_refs, 2),
                   'refs_per_query_distribution':
                       {str(kk): vv for kk, vv in sorted(Counter(nref).items())},
                   'absolute': {cid: {'recall': float(scored[cid]['r'].mean()),
                                      'mrr': float(scored[cid]['mrr'].mean()),
                                      'mean_hits_at_10': float(scored[cid]['hits'].mean())}
                                for cid in d['classes']},
                   'contrasts': {}}
            for a, b, lbl in HEADLINE[corpus]:
                dr = scored[a]['r'] - scored[b]['r']
                dh = scored[a]['hits'] - scored[b]['hits']
                p_q, lo_q, hi_q = paired_bootstrap(dr, idxq)
                p_c, lo_c, hi_c = cluster_bootstrap(dr, d['cluster_ids'],
                                                    d['n_clusters'], rng)
                p_s, kpos, mnz = sign_test_p(dr)
                if dr.mean() < 0:
                    lo_q, hi_q = -hi_q, -lo_q
                    lo_c, hi_c = -hi_c, -lo_c
                rec['contrasts'][lbl] = {
                    'a': a, 'b': b,
                    'delta_recall': float(dr.mean()),
                    'delta_hits_at_10': float(dh.mean()),
                    'ci95_query': [lo_q, hi_q], 'ci95_cluster': [lo_c, hi_c],
                    'p_query': p_q, 'p_cluster': p_c, 'p_sign': p_s,
                    'sign_wins': kpos, 'sign_n': mnz,
                    'survive_query': bool(p_q < alpha),
                    'survive_cluster': bool(p_c < alpha),
                    'survive_sign': bool(p_s < alpha),
                    'n_improved': int((dr > 0).sum()),
                    'n_unchanged': int((dr == 0).sum()),
                    'n_worse': int((dr < 0).sum()),
                }
            rows.append(rec)
            print(f"\n  --- K={depth}  (mean refs/query = {mean_refs:.2f}) ---")
            for cid in d['classes']:
                ab = rec['absolute'][cid]
                print(f"      {cid} {d['classes'][cid]:<18} R@10={ab['recall']:.4f} "
                      f"MRR={ab['mrr']:.4f} hits@10={ab['mean_hits_at_10']:.2f}")
            for _, _, lbl in HEADLINE[corpus]:
                c = rec['contrasts'][lbl]
                print(f"      {lbl:<38} dR={c['delta_recall']:+.4f} "
                      f"dHits={c['delta_hits_at_10']:+.3f} "
                      f"cCI=[{c['ci95_cluster'][0]:+.4f},{c['ci95_cluster'][1]:+.4f}] "
                      f"q/c/s={int(c['survive_query'])}{int(c['survive_cluster'])}"
                      f"{int(c['survive_sign'])}  win/tie/loss="
                      f"{c['n_improved']}/{c['n_unchanged']}/{c['n_worse']}")
        report['corpora'][corpus] = {
            'n_queries': n, 'n_classes': k, 'bonferroni_alpha': alpha,
            'n_clusters': d['n_clusters'], 'classes': dict(d['classes']),
            'by_depth': rows}

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote evaluation/{os.path.basename(OUT)}")


if __name__ == '__main__':
    main()
