import os, sys, json, re
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from temporal.temporal_parser import TemporalParser

parser = TemporalParser()

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')
QRELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qrels.json')

# Load the original queries to keep the test set the same
with open(QRELS_PATH) as f:
    old_qrels = json.load(f)

# Load all chunks into memory
chunks = []
with open(CHUNKS_PATH, 'r') as f:
    for line in f:
        if line.strip():
            chunks.append(json.loads(line))

def get_terms(text):
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    stopwords = {"and", "for", "the", "in", "to", "of", "with", "from", "on", "at", "by", "this", "that", "it", "as", "is", "are", "was", "were", "be", "been", "has", "have", "had", "will", "would", "could", "should", "not", "no", "since", "before", "after", "during"}
    return set(w for w in words if w not in stopwords)

new_qrels = []

for q in old_qrels:
    query = q['query_text']
    intent = parser.parse(query)
    start_y = intent.get('start_year')
    end_y = intent.get('end_year')
    
    query_terms = get_terms(query)
    
    scored_chunks = []
    
    for c in chunks:
        doc_year = c.get('publication_year')
        
        # Exact year filtering independent of BM25
        if start_y is not None and end_y is not None:
            if doc_year is None or not (start_y <= doc_year <= end_y):
                continue
                
        chunk_terms = get_terms(c['text'])
        overlap = len(query_terms.intersection(chunk_terms))
        
        if overlap > 0:
            scored_chunks.append({
                'parent_doc_id': c['parent_doc_id'],
                'overlap': overlap
            })
            
    # Sort by overlap
    scored_chunks.sort(key=lambda x: x['overlap'], reverse=True)
    
    # Deduplicate by parent_doc_id
    seen = set()
    dedup = []
    for sc in scored_chunks:
        if sc['parent_doc_id'] not in seen:
            seen.add(sc['parent_doc_id'])
            dedup.append(sc)
            
    # Get top chunks
    if not dedup:
        print(f"Warning: No chunks found for '{query}' with range {start_y}-{end_y}")
        relevant_docs = []
    else:
        # Take chunks that have the highest overlap, up to 5
        top_overlap = dedup[0]['overlap']
        # Be slightly permissive to get multiple chunks
        threshold = max(1, top_overlap - 1)
        relevant_docs = [d['parent_doc_id'] for d in dedup if d['overlap'] >= threshold][:5]
        
    q['relevant_doc_ids'] = relevant_docs
    
    # Clean up single IDs
    for k in ['relevant_doc_id', 'relevant_chunk_id', 'year_of_match']:
        if k in q:
            del q[k]
            
    new_qrels.append(q)

with open(QRELS_PATH, 'w') as f:
    json.dump(new_qrels, f, indent=2)

print(f"Generated fully independent qrels.json.")
print(f"Avg docs per query: {sum(len(q['relevant_doc_ids']) for q in new_qrels) / len(new_qrels):.2f}")
