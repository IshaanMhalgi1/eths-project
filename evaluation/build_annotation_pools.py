#!/usr/bin/env python
"""
Build candidate pools for human annotation.
- 10 queries selected from existing qrels
- For each query, union of top-20 from BM25, Dense, Hybrid
- Deduplicated using canonical mapping
- Output: annotation-ready JSON with document text, metadata, and empty relevance fields
"""
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25, DEFAULT_INDEX, SMALL_INDEX
from retrieval.dense import search_dense
from retrieval.hybrid import search_hybrid
from api.canonical_docs import get_canonical

# 10 representative queries spanning both corpora and temporal types.
#
# v2 CHANGES (blind re-run):
#   - q5  -> q10  "tax land assessment 1804 to 1808"   (q5 was previously spot-checked)
#   - q21 -> q33  "Erie Canal completion 1825"         (q21 was previously spot-checked
#                                                       AND drew chance-level annotator
#                                                       agreement, kappa = -0.012)
#   Both replacements verified never-before-seen in any report or analysis script.
#   Candidate order is randomized in the UI and retriever scores are stripped
#   from this file entirely (blind annotation).
SELECTED_QUERIES = [
    # Small corpus (1800-1810) specific - temporal explicit
    {"query_id": "q10", "query_text": "tax land assessment 1804 to 1808", "temporal_type": "range", "corpus": "small"},
    {"query_id": "q9", "query_text": "political news congress 1800 to 1805", "temporal_type": "range", "corpus": "small"},
    {"query_id": "q14", "query_text": "shipping port arrivals after 1805", "temporal_type": "before_after", "corpus": "small"},
    
    # Expanded corpus (1800-1900) specific - temporal explicit  
    {"query_id": "q33", "query_text": "Erie Canal completion 1825", "temporal_type": "explicit_year", "corpus": "expanded"},
    {"query_id": "q31", "query_text": "Monroe Doctrine foreign policy 1823", "temporal_type": "explicit_year", "corpus": "expanded"},
    {"query_id": "q42", "query_text": "Trail of Tears Cherokee removal 1838", "temporal_type": "explicit_year", "corpus": "expanded"},
    {"query_id": "q71", "query_text": "Civil War Fort Sumter 1861", "temporal_type": "explicit_year", "corpus": "expanded"},
    
    # Topical/implicit - work on both corpora
    {"query_id": "q1", "query_text": "strayed horse reward advertisement", "temporal_type": "none", "corpus": "both"},
    {"query_id": "q3", "query_text": "runaway slave reward advertisement", "temporal_type": "none", "corpus": "both"},
    {"query_id": "q101", "query_text": "agricultural fair prize livestock", "temporal_type": "none", "corpus": "both"},
]

TOP_K = 20

# (query_id, canonical_parent_doc_id) -> {retriever: score}, for the
# post-hoc scored sidecar only. Populated during pool construction.
_SCORE_MAP = {}

def get_index_for_query(corpus):
    """Return appropriate index name for a query's corpus."""
    if corpus == "small":
        return SMALL_INDEX
    else:
        return DEFAULT_INDEX  # expanded corpus uses main index

def search_all(query_text, index_name, top_k=TOP_K):
    """Get top-K from BM25, Dense, Hybrid."""
    bm25_results = search_bm25(query_text, size=top_k, index_name=index_name)
    dense_results = search_dense(query_text, size=top_k, index_name=index_name)
    # Use alpha=0.7 for small, 1.0 for expanded (per RRF sweep results)
    alpha = 0.7 if index_name == SMALL_INDEX else 1.0
    hybrid_results = search_hybrid(query_text, size=top_k, alpha=alpha, index_name=index_name)
    return bm25_results, dense_results, hybrid_results

def deduplicate_candidates(all_results):
    """Deduplicate by canonical parent_doc_id.

    v2 FIX: the previous version deduplicated the MERGED list and then took
    the top-20 by raw score. Because BM25 scores (~5-14) sort above Dense
    (~0.7) which sort above Hybrid RRF (~0.016), the "top-20 of the merged
    list" was in practice BM25's top-20 — Dense contributed almost nothing to
    the judged pool. That biased every Dense measurement in the v1 pilot
    against Dense, since a document outside the pool can never be found
    relevant.

    The correct TREC pooling construction is the UNION of each retriever's
    top-K, not the top-K of the merged ranking. Membership in the pool must
    not depend on a retriever's score scale.
    """
    seen = {}
    for r in all_results:
        canonical_pid = get_canonical(r['parent_doc_id'])
        if canonical_pid not in seen:
            r['canonical_parent_doc_id'] = canonical_pid
            seen[canonical_pid] = r
        else:
            # keep the record but remember the best score seen
            if r['score'] > seen[canonical_pid]['score']:
                r['canonical_parent_doc_id'] = canonical_pid
                seen[canonical_pid] = r
    return list(seen.values())

def build_annotation_pool(query_info):
    """Build candidate pool for one query."""
    index_name = get_index_for_query(query_info['corpus'])
    bm25, dense, hybrid = search_all(query_info['query_text'], index_name)
    
    # UNION of each retriever's top-K, then dedup. No global score sort —
    # see deduplicate_candidates() for why that is invalid across retrievers
    # with different score scales.
    all_results = bm25 + dense + hybrid
    deduped = deduplicate_candidates(all_results)
    
    # Deterministic display order that does not encode any retriever's ranking:
    # round-robin across the three retrievers' contributions after dedup.
    sources = [bm25, dense, hybrid]
    queues = [[r['parent_doc_id'] for r in src] for src in sources]
    cursors = [0, 0, 0]
    order, emitted = [], set()
    while len(order) < len(deduped):
        progressed = False
        for si in range(3):
            while cursors[si] < len(queues[si]):
                cpid = get_canonical(queues[si][cursors[si]])
                cursors[si] += 1
                if cpid not in emitted:
                    emitted.add(cpid)
                    order.append(cpid)
                    progressed = True
                    break
        if not progressed:
            break
    by_canon = {get_canonical(r['parent_doc_id']): r for r in deduped}
    deduped = [by_canon[c] for c in order if c in by_canon]
    
    # Format for annotation
    candidates = []
    for i, r in enumerate(deduped):  # full union, no cap
        _SCORE_MAP[(query_info['query_id'], r['canonical_parent_doc_id'])] = {
            "bm25": next((x['score'] for x in bm25 if get_canonical(x['parent_doc_id']) == r['canonical_parent_doc_id']), None),
            "dense": next((x['score'] for x in dense if get_canonical(x['parent_doc_id']) == r['canonical_parent_doc_id']), None),
            "hybrid": next((x['score'] for x in hybrid if get_canonical(x['parent_doc_id']) == r['canonical_parent_doc_id']), None),
        }
        candidates.append({
            "candidate_id": f"{query_info['query_id']}_cand_{i}",
            "canonical_parent_doc_id": r['canonical_parent_doc_id'],
            "original_parent_doc_id": r['parent_doc_id'],
            "chunk_id": r['chunk_id'],
            # NOTE: retriever_scores are computed but deliberately NOT written to
            # the annotation file. In the v1 pilot they were displayed to
            # annotators, which is a plausible anchoring hazard and a known
            # confound on that run. They are retained in the non-blind copy
            # (annotation_pools_with_scores.json) for post-hoc analysis only.
            "text": r['text'],
            "metadata": {
                "publication_year": r.get('publication_year'),
                "historical_start": r.get('historical_start'),
                "historical_end": r.get('historical_end'),
                "historical_period": r.get('historical_period'),
                "location": r.get('location')
            },
            # Empty fields for annotators to fill
            "relevance_annotator1": None,
            "relevance_annotator2": None,
            "relevance_final": None,
            "notes_annotator1": "",
            "notes_annotator2": ""
        })
    
    return {
        "query_id": query_info['query_id'],
        "query_text": query_info['query_text'],
        "temporal_type": query_info['temporal_type'],
        "corpus": query_info['corpus'],
        "num_candidates": len(candidates),
        "candidates": candidates
    }

def main():
    print("Building candidate pools for human annotation...")
    print(f"Selected queries: {len(SELECTED_QUERIES)}")
    
    pools = []
    for q in SELECTED_QUERIES:
        print(f"  Processing {q['query_id']}: {q['query_text']} [{q['corpus']}]")
        pool = build_annotation_pool(q)
        pools.append(pool)
        print(f"    -> {pool['num_candidates']} unique candidates (after dedup)")
    
    # Save annotation pools
    output = {
        "version": "1.0",
        "description": "Human annotation candidate pools for ETHS evaluation. 10 queries, top-20 union of BM25/Dense/Hybrid, deduplicated via canonical mapping.",
        "annotator_guidelines_ref": "ANNOTATOR_GUIDELINES.md",
        "queries": pools
    }
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'annotation_pools.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {output_path}")

    # Post-hoc sidecar WITH retriever scores. Never served to annotators;
    # used only after annotation is complete, to check whether a ranker's
    # documents were disproportionately judged relevant (i.e. whether
    # retriever identity predicts human relevance).
    scored = {
        "version": "2.0-scored",
        "_warning": "CONTAINS RETRIEVER SCORES. Do not give this file to annotators.",
        "queries": []
    }
    for pool in pools:
        scored["queries"].append({
            "query_id": pool['query_id'],
            "query_text": pool['query_text'],
            "candidates": [
                {"canonical_parent_doc_id": c['canonical_parent_doc_id']}
                for c in pool['candidates']
            ]
        })
    # Re-derive scores by re-running retrieval would be wasteful; instead
    # recover them from the in-memory dedup step by rebuilding the map.
    # Simpler: store the per-candidate best-score provenance recorded above.
    for pool, sp in zip(pools, scored["queries"]):
        for c, sc in zip(pool['candidates'], sp["candidates"]):
            sc["retriever_scores"] = _SCORE_MAP.get(
                (pool['query_id'], c['canonical_parent_doc_id']))

    scored_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                               'annotation_pools_with_scores.json')
    with open(scored_path, 'w') as f:
        json.dump(scored, f, indent=2)
    print(f"Scored sidecar (NOT for annotators) saved to {scored_path}")
    
    # Also create a simplified annotation sheet per query (one file per query for easier distribution)
    annotation_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'annotation')
    os.makedirs(annotation_dir, exist_ok=True)
    
    for pool in pools:
        qid = pool['query_id']
        sheet = {
            "query_id": pool['query_id'],
            "query_text": pool['query_text'],
            "temporal_type": pool['temporal_type'],
            "corpus": pool['corpus'],
            "candidates": pool['candidates']
        }
        sheet_path = os.path.join(annotation_dir, f"{qid}_annotation.json")
        with open(sheet_path, 'w') as f:
            json.dump(sheet, f, indent=2)
    
    print(f"Individual annotation sheets saved to {annotation_dir}/")

if __name__ == '__main__':
    main()