import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from evaluation.ablation_rankers import get_ranker, CORPUS_CONFIG

for corpus in ['small', 'expanded']:
    print("=== " + CORPUS_CONFIG[corpus]["name"] + " ===")
    for name in ['BM25', 'Dense', 'Hybrid', 'Hybrid+Temporal', 'Hybrid+Metadata', 'Final', 'BM25+Temporal', 'Dense+Temporal', 'TemporalOnly', 'MetadataOnly']:
        fn = get_ranker(name, corpus)
        res = fn('real estate sales in 1805')
        top = res[0]['parent_doc_id'] if res else 'none'
        print("  " + name + ": " + str(len(res)) + " results, top=" + top)