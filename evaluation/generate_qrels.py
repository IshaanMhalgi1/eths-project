import os
import json
from datasets import load_dataset
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')

with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

MAX_CHUNKS = cfg.get('max_chunks', 2000)

def main():
    dataset = load_dataset('Bhawna/ChroniclingAmericaQA', split='train')
    subset = dataset.select(range(min(MAX_CHUNKS, len(dataset))))
    
    qrels = []
    # Just take the first 20 distinct questions as our evaluation set
    count = 0
    seen_ids = set()
    for doc in subset:
        qid = str(doc.get('query_id'))
        if qid in seen_ids:
            continue
        seen_ids.add(qid)
        qrels.append({
            "query_id": qid,
            "query_text": doc.get('question'),
            "relevant_doc_id": qid
        })
        count += 1
        if count >= 20:
            break
            
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(qrels, f, indent=2)
    print(f"Generated {len(qrels)} test queries in {OUT_PATH}")

if __name__ == '__main__':
    main()
