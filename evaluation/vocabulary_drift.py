"""
Vocabulary Drift Experiment — Cross-Decade Retrieval Quality

Tests whether queries from one decade retrieve relevant documents from another decade,
measuring the effect of historical language change on retrieval performance.
"""
import os
import sys
import json
import numpy as np
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense
from ranking.shared_hybrid import get_hybrid_candidates, clear_hybrid_cache
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

# Load qrels (expanded corpus, 90 queries)
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

print("Queries per decade:")
for d in sorted(queries_by_decade):
    print(f"  {d}s: {len(queries_by_decade[d])} queries")

# Load all chunks to get publication year distribution
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')
chunks = []
with open(CHUNKS_PATH, 'r') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

# Corpus stats by decade
corpus_by_decade = defaultdict(int)
for c in chunks:
    yr = c.get('publication_year')
    if yr and 1800 <= yr < 1900:
        decade = (yr // 10) * 10
        corpus_by_decade[decade] += 1

print("\nCorpus chunks per decade:")
for d in sorted(corpus_by_decade):
    print(f"  {d}s: {corpus_by_decade[d]}")

def evaluate_on_corpus_subset(query_text, true_docs, target_decade=None, method='bm25', k=10):
    """
    Evaluate a query against documents from a specific decade (or all).
    """
    if method == 'bm25':
        # For BM25, we'd need to filter results post-retrieval
        # Simpler: use the full search and filter
        results = search_bm25(query_text, size=50)
    elif method == 'hybrid':
        clear_hybrid_cache()
        results = get_hybrid_candidates(query_text)
    elif method == 'dense':
        results = search_dense(query_text, size=50)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Filter by target decade if specified
    if target_decade is not None:
        filtered = []
        for r in results:
            yr = r.get('publication_year')
            if yr and (target_decade <= yr < target_decade + 10):
                filtered.append(r)
        results = filtered
    
    # Deduplicate by parent_doc_id
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


# Main experiment: For each query decade, evaluate against same-decade vs other-decade docs
print("\n" + "="*80)
print("VOCABULARY DRIFT EXPERIMENT: Cross-Decade Retrieval")
print("="*80)

methods = ['bm25', 'hybrid', 'dense']
decades = sorted(queries_by_decade.keys())

for method in methods:
    print(f"\n--- Method: {method.upper()} ---")
    
    for query_decade in decades:
        queries = queries_by_decade[query_decade]
        if len(queries) < 3:
            continue
            
        print(f"\n  Query decade: {query_decade}s ({len(queries)} queries)")
        
        # Same-decade retrieval
        same_decade_recalls, same_decade_mrrs = [], []
        for q in queries:
            res = evaluate_on_corpus_subset(
                q['query_text'], q['relevant_doc_ids'], 
                target_decade=query_decade, method=method
            )
            same_decade_recalls.append(res['Recall@10'])
            same_decade_mrrs.append(res['MRR'])
        
        # Cross-decade retrieval (all other decades)
        cross_decade_recalls, cross_decade_mrrs = [], []
        for q in queries:
            res = evaluate_on_corpus_subset(
                q['query_text'], q['relevant_doc_ids'], 
                target_decade=None, method=method  # full corpus
            )
            cross_decade_recalls.append(res['Recall@10'])
            cross_decade_mrrs.append(res['MRR'])
        
        print(f"    Same-decade:  Recall@10={np.mean(same_decade_recalls):.3f}, MRR={np.mean(same_decade_mrrs):.3f}")
        print(f"    Cross-decade: Recall@10={np.mean(cross_decade_recalls):.3f}, MRR={np.mean(cross_decade_mrrs):.3f}")

# Now test specific decade pairs
print("\n" + "="*80)
print("SPECIFIC DECADE PAIRS (BM25)")
print("="*80)

for query_decade in decades:
    queries = queries_by_decade[query_decade]
    if len(queries) < 3:
        continue
    
    for target_decade in sorted(corpus_by_decade.keys()):
        if target_decade == query_decade:
            continue
        if corpus_by_decade[target_decade] < 1000:
            continue  # skip sparse decades
            
        pair_recalls, pair_mrrs = [], []
        for q in queries:
            res = evaluate_on_corpus_subset(
                q['query_text'], q['relevant_doc_ids'],
                target_decade=target_decade, method='bm25'
            )
            pair_recalls.append(res['Recall@10'])
            pair_mrrs.append(res['MRR'])
        
        if len(pair_recalls) > 0:
            print("  {}s query -> {}s docs: Recall@10={:.3f}, MRR={:.3f} (n={})".format(
                query_decade, target_decade, np.mean(pair_recalls), np.mean(pair_mrrs), len(pair_recalls)))

# Summary: drift magnitude
print("\n" + "="*80)
print("DRIFT SUMMARY: Same-decade vs Full-corpus (BM25)")
print("="*80)

for query_decade in decades:
    queries = queries_by_decade[query_decade]
    if len(queries) < 3:
        continue
    
    same_recalls, full_recalls = [], []
    for q in queries:
        r1 = evaluate_on_corpus_subset(q['query_text'], q['relevant_doc_ids'], target_decade=query_decade, method='bm25')
        r2 = evaluate_on_corpus_subset(q['query_text'], q['relevant_doc_ids'], target_decade=None, method='bm25')
        same_recalls.append(r1['Recall@10'])
        full_recalls.append(r2['Recall@10'])
    
    diff = np.mean(full_recalls) - np.mean(same_recalls)
    print("  {}s: Same={:.3f}, Full={:.3f}, Delta={:+.3f}".format(
        query_decade, np.mean(same_recalls), np.mean(full_recalls), diff))