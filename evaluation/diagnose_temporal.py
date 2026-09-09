import os, sys, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.hybrid import search_hybrid
from ranking.temporal_ranker import search_temporal, parser

QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')
with open(QRELS_PATH) as f:
    qrels = json.load(f)

print("Diagnosing Temporal Ranking Regression...")
print("=" * 70)

for q in qrels:
    query = q['query_text']
    expected_doc = q['relevant_doc_id']
    
    # 1. Run hybrid
    hybrid_res = search_hybrid(query, size=10, alpha=0.5)
    if not hybrid_res:
        continue
    hybrid_top_doc = hybrid_res[0]['parent_doc_id']
    
    if hybrid_top_doc == expected_doc:
        # Hybrid got it at #1. Let's see what temporal did.
        temp_res = search_temporal(query, size=10, alpha_hybrid=0.7, beta_temporal=0.3)
        temp_ranks = [r['parent_doc_id'] for r in temp_res]
        
        if expected_doc not in temp_ranks or temp_ranks.index(expected_doc) > 0:
            print(f"\nQuery: '{query}'")
            parsed = parser.parse(query)
            print(f"  Parsed box: {parsed.get('start_year')} to {parsed.get('end_year')}")
            
            # Get expected doc's temporal score
            temp_res_large = search_temporal(query, size=100, alpha_hybrid=0.7, beta_temporal=0.3)
            expected_data = next((r for r in temp_res_large if r['parent_doc_id'] == expected_doc), None)
            
            print(f"  Expected Doc ({expected_doc}):")
            if expected_data:
                idx = [r['parent_doc_id'] for r in temp_res_large].index(expected_doc)
                print(f"    New Rank: {idx+1}")
                print(f"    Doc dates: {expected_data.get('historical_start')} to {expected_data.get('historical_end')}")
                print(f"    Temp Score: {expected_data.get('temporal_score')} ({expected_data.get('temporal_explanation')})")
                print(f"    Hybrid base score: {expected_data.get('score'):.4f} | Final score: {expected_data.get('final_score'):.4f}")
            else:
                print("    [Fell out of top 100]")
                
            print("  Docs ranked above it by Temporal Ranker:")
            limit = temp_ranks.index(expected_doc) if expected_doc in temp_ranks else min(3, len(temp_res))
            for i in range(limit):
                r = temp_res[i]
                print(f"    #{i+1} {r['parent_doc_id']} | TempScore: {r.get('temporal_score')} ({r.get('temporal_explanation')}) | Dates: {r.get('historical_start')}-{r.get('historical_end')} | Hybrid base: {r.get('score'):.4f}")
            print("-" * 70)
