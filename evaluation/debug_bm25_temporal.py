import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.bm25 import search_bm25, SMALL_INDEX
from ranking.temporal_ranker import calculate_temporal_score
from temporal.temporal_parser import TemporalParser
import json

parser = TemporalParser()

# Load small qrels
with open('data/qrels_small.json') as f:
    qrels = json.load(f)

# Find a temporal query
temporal_query = None
for q in qrels:
    intent = parser.parse(q['query_text'])
    if intent.get('start_year') and intent.get('end_year'):
        temporal_query = q
        break

print("Query:", temporal_query['query_text'])
print("Query ID:", temporal_query['query_id'])
intent = parser.parse(temporal_query['query_text'])
print("Parsed:", intent)
print("Relevant docs:", temporal_query['relevant_doc_ids'])

# Get BM25 results
bm25_res = search_bm25(temporal_query['query_text'], size=50, index_name=SMALL_INDEX)
print("\n=== BM25 Top 10 ===")
for i, r in enumerate(bm25_res[:10]):
    yr = r.get('publication_year')
    start = r.get('historical_start')
    end = r.get('historical_end')
    t_score, t_exp = calculate_temporal_score(start, end, intent['start_year'], intent['end_year'])
    print(f"  {i+1}. {r['parent_doc_id']} (yr={yr}, hist=[{start},{end}], temp={t_score:.1f}/{t_exp}): {r['score']:.6f}")

# Now run BM25+Temporal ranker
from evaluation.ablation_rankers import rank_bm25, rank_bm25_temporal

bm25_ranked = rank_bm25(temporal_query['query_text'], size=10, index_name=SMALL_INDEX)
bm25_temp_ranked = rank_bm25_temporal(temporal_query['query_text'], size=10, index_name=SMALL_INDEX)

print("\n=== BM25 Ranked ===")
for i, r in enumerate(bm25_ranked[:10]):
    print(f"  {i+1}. {r['parent_doc_id']} final={r['final_score']:.6f} bm25={r['score']:.6f}")

print("\n=== BM25+Temporal Ranked ===")
for i, r in enumerate(bm25_temp_ranked[:10]):
    print(f"  {i+1}. {r['parent_doc_id']} final={r['final_score']:.6f} bm25={r['bm25_score']:.6f} temp={r.get('temporal_score', 0):.6f}")

# Check per-query scores
print("\n=== Per-query BM25 vs BM25+Temporal scores ===")
for q in qrels[:5]:
    intent = parser.parse(q['query_text'])
    if not (intent.get('start_year') and intent.get('end_year')):
        continue
    b = rank_bm25(q['query_text'], size=10, index_name=SMALL_INDEX)
    bt = rank_bm25_temporal(q['query_text'], size=10, index_name=SMALL_INDEX)
    print(f"\n{q['query_id']}: {q['query_text'][:50]}...")
    print(f"  BM25 top: {[r['parent_doc_id'] + '(' + str(r['final_score'])[:8] + ')' for r in b[:3]]}")
    print(f"  BM25+T top: {[r['parent_doc_id'] + '(' + str(r['final_score'])[:8] + ')' for r in bt[:3]]}")
    # Check if order changed
    b_ids = [r['parent_doc_id'] for r in b]
    bt_ids = [r['parent_doc_id'] for r in bt]
    if b_ids != bt_ids:
        print("  *** ORDER CHANGED ***")
    else:
        print("  Same order")