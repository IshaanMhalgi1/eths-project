import os
import sys
import json
from collections import defaultdict, Counter
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense
from ranking.shared_hybrid import get_hybrid_candidates, clear_hybrid_cache
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

# Load qrels
QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
with open(QRELS_PATH) as f:
    qrels = json.load(f)
qrels = [q for q in qrels if q.get('relevant_doc_ids')]

# Group queries by their temporal intent (start_year)
queries_by_decade = defaultdict(list)
for q in qrels:
    intent = parser.parse(q['query_text'])
    start_y = intent.get('start_year')
    if start_y:
        decade = (start_y // 10) * 10
        queries_by_decade[decade].append(q)

# Load chunks for corpus stats
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')
chunks = []
with open(CHUNKS_PATH, 'r') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

corpus_by_decade = defaultdict(int)
for c in chunks:
    yr = c.get('publication_year')
    if yr and 1800 <= yr < 1900:
        decade = (yr // 10) * 10
        corpus_by_decade[decade] += 1

def evaluate_on_corpus_subset(query_text, true_docs, target_decade=None, method='bm25', k=10):
    if method == 'bm25':
        results = search_bm25(query_text, size=50)
    elif method == 'hybrid':
        clear_hybrid_cache()
        results = get_hybrid_candidates(query_text)
    elif method == 'dense':
        results = search_dense(query_text, size=50)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if target_decade is not None:
        filtered = []
        for r in results:
            yr = r.get('publication_year')
            if yr and (target_decade <= yr < target_decade + 10):
                filtered.append(r)
        results = filtered
    
    seen = set()
    dedup = []
    for r in results:
        pid = r['parent_doc_id']
        if pid not in seen:
            seen.add(pid)
            dedup.append(pid)
        if len(dedup) == k:
            break
    
    hits = [1 if d in true_docs else 0 for d in dedup]
    n_rel = max(len(true_docs), 1)
    recall = sum(hits) / n_rel
    
    mrr = 0
    for rank, d in enumerate(dedup):
        if d in true_docs:
            mrr = 1.0 / (rank + 1)
            break
    
    return {'Recall@10': recall, 'MRR': mrr, 'n_results': len(results)}

print("=" * 100)
print("TABLE 1: Same-Decade vs Cross-Decade (Full Corpus) Retrieval by Method")
print("=" * 100)

methods = ['bm25', 'hybrid', 'dense']
decades = sorted(queries_by_decade.keys())

# Header
header = f"{'Method':<10} | {'Query Decade':<12} | {'Same-Decade R@10':>16} | {'Same-Decade MRR':>15} | {'Full-Corpus R@10':>16} | {'Full-Corpus MRR':>15} | {'Delta R@10':>12}"
print(header)
print("-" * len(header))

for method in methods:
    for query_decade in decades:
        queries = queries_by_decade[query_decade]
        if len(queries) < 3:
            continue
        
        same_decade_recalls, same_decade_mrrs = [], []
        cross_decade_recalls, cross_decade_mrrs = [], []
        
        for q in queries:
            res_same = evaluate_on_corpus_subset(q['query_text'], q['relevant_doc_ids'], target_decade=query_decade, method=method)
            res_cross = evaluate_on_corpus_subset(q['query_text'], q['relevant_doc_ids'], target_decade=None, method=method)
            same_decade_recalls.append(res_same['Recall@10'])
            same_decade_mrrs.append(res_same['MRR'])
            cross_decade_recalls.append(res_cross['Recall@10'])
            cross_decade_mrrs.append(res_cross['MRR'])
        
        mean_same_r = np.mean(same_decade_recalls)
        mean_same_mrr = np.mean(same_decade_mrrs)
        mean_cross_r = np.mean(cross_decade_recalls)
        mean_cross_mrr = np.mean(cross_decade_mrrs)
        delta = mean_cross_r - mean_same_r
        
        row = f"{method.upper():<10} | {query_decade}s{'':<6} | {mean_same_r:>16.3f} | {mean_same_mrr:>15.3f} | {mean_cross_r:>16.3f} | {mean_cross_mrr:>15.3f} | {delta:>+12.3f}"
        print(row)
    if method != methods[-1]:
        print()

print()
print("=" * 100)
print("TABLE 2: Specific Decade-Pair Retrieval (BM25) - Query Decade -> Target Corpus Decade")
print("=" * 100)

header2 = f"{'Query Decade':<14} | {'Target Decade':<14} | {'Recall@10':>11} | {'MRR':>7} | {'Queries':>8}"
print(header2)
print("-" * len(header2))

for query_decade in decades:
    queries = queries_by_decade[query_decade]
    if len(queries) < 3:
        continue
    
    for target_decade in sorted(corpus_by_decade.keys()):
        if target_decade == query_decade:
            continue
        if corpus_by_decade[target_decade] < 1000:
            continue
            
        pair_recalls, pair_mrrs = [], []
        for q in queries:
            res = evaluate_on_corpus_subset(q['query_text'], q['relevant_doc_ids'], target_decade=target_decade, method='bm25')
            pair_recalls.append(res['Recall@10'])
            pair_mrrs.append(res['MRR'])
        
        if len(pair_recalls) > 0:
            mean_r = np.mean(pair_recalls)
            mean_mrr = np.mean(pair_mrrs)
            row = f"{query_decade}s{'':<10} | {target_decade}s{'':<10} | {mean_r:>11.3f} | {mean_mrr:>7.3f} | {len(pair_recalls):>8}"
            print(row)

print()
print("=" * 100)
print("TABLE 3: Drift Summary - Same-Decade vs Full-Corpus (BM25)")
print("=" * 100)

header3 = f"{'Query Decade':<14} | {'Same-Decade R@10':>16} | {'Full-Corpus R@10':>16} | {'Delta (Full-Same)':>18} | {'Same-Decade MRR':>15} | {'Full-Corpus MRR':>15}"
print(header3)
print("-" * len(header3))

for query_decade in decades:
    queries = queries_by_decade[query_decade]
    if len(queries) < 3:
        continue
    
    same_recalls, full_recalls = [], []
    same_mrrs, full_mrrs = [], []
    for q in queries:
        r1 = evaluate_on_corpus_subset(q['query_text'], q['relevant_doc_ids'], target_decade=query_decade, method='bm25')
        r2 = evaluate_on_corpus_subset(q['query_text'], q['relevant_doc_ids'], target_decade=None, method='bm25')
        same_recalls.append(r1['Recall@10'])
        full_recalls.append(r2['Recall@10'])
        same_mrrs.append(r1['MRR'])
        full_mrrs.append(r2['MRR'])
    
    diff_r = np.mean(full_recalls) - np.mean(same_recalls)
    diff_mrr = np.mean(full_mrrs) - np.mean(same_mrrs)
    row = f"{query_decade}s{'':<10} | {np.mean(same_recalls):>16.3f} | {np.mean(full_recalls):>16.3f} | {diff_r:>+18.3f} | {np.mean(same_mrrs):>15.3f} | {np.mean(full_mrrs):>15.3f}"
    print(row)

print()
print("=" * 100)
print("TABLE 4: Query-Decade vs Relevant-Doc-Decade Alignment (from qrels construction)")
print("=" * 100)

# Load chunks for year lookup
chunk_year = {}
with open(CHUNKS_PATH, 'r') as f:
    for line in f:
        if line.strip():
            c = json.loads(line)
            chunk_year[c['parent_doc_id']] = c.get('publication_year')

header4 = f"{'Query Decade':<14} | {'Queries':>7} | {'Match Decade':>12} | {'Mismatch Decade':>15} | {'Top Rel Decade Dist':<40}"
print(header4)
print("-" * len(header4))

for qdec in sorted(queries_by_decade):
    items = queries_by_decade[qdec]
    match_count = 0
    mismatch_count = 0
    decade_dists = Counter()
    
    for q in items:
        rel_decades = []
        for doc_id in q['relevant_doc_ids']:
            yr = chunk_year.get(doc_id)
            if yr and 1800 <= yr < 1900:
                rel_decades.append((yr // 10) * 10)
        if rel_decades:
            dec_counts = Counter(rel_decades)
            top_dec = dec_counts.most_common(1)[0][0]
            if top_dec == qdec:
                match_count += 1
            else:
                mismatch_count += 1
            decade_dists[top_dec] += 1
    
    dist_str = ", ".join(f"{d}s:{c}" for d, c in sorted(decade_dists.items()))
    row = f"{qdec}s{'':<10} | {len(items):>7} | {match_count:>12} | {mismatch_count:>15} | {dist_str:<40}"
    print(row)