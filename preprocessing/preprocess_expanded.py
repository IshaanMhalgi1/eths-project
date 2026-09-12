import os
import json
import yaml
import uuid
from typing import List, Dict
from collections import defaultdict

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

EMBEDDING_MODEL = cfg.get('embedding_model', 'all-MiniLM-L6-v2')
MAX_CHUNKS = cfg.get('max_chunks', 50000)
CORPUS_START_YEAR = cfg.get('corpus_start_year', 1800)
CORPUS_END_YEAR = cfg.get('corpus_end_year', 1900)

model = SentenceTransformer(EMBEDDING_MODEL)

def clean_text(text: str) -> str:
    return " ".join(text.split())

def chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    cur_len = 0
    for sent in sentences:
        tok_len = len(sent.split())
        if cur_len + tok_len > max_tokens:
            if current:
                chunks.append(' '.join(current))
            current = [sent]
            cur_len = tok_len
        else:
            current.append(sent)
            cur_len += tok_len
    if current:
        chunks.append(' '.join(current))
    return chunks

def process_document(doc: Dict) -> List[Dict]:
    parent_id = str(doc.get('query_id', uuid.uuid4()))
    raw_text = clean_text(doc.get('context') or '')
    
    pub_date = doc.get('publication_date')
    pub_year = None
    if pub_date:
        try:
            pub_year = int(pub_date.split('-')[0])
        except (ValueError, AttributeError):
            pass

    chunks = chunk_text(raw_text)
    records = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{parent_id}_{i}"
        record = {
            "chunk_id": chunk_id,
            "parent_doc_id": parent_id,
            "title": None,
            "author": None,
            "text": chunk,
            "publication_year": pub_year,
            "historical_start": pub_year,
            "historical_end": pub_year,
            "location": None,
            "historical_period": None,
            "topic": None,
            "document_type": None,
            "source": "Chronicling America",
            "provenance": doc.get('url', ''),
        }
        records.append(record)
    return records

def main():
    print("Loading ChroniclingAmericaQA dataset...")
    dataset = load_dataset('Bhawna/ChroniclingAmericaQA', split='train')
    
    # Group documents by decade
    docs_by_decade = defaultdict(list)
    for doc in dataset:
        pub_date = doc.get('publication_date')
        if pub_date:
            try:
                yr = int(pub_date.split('-')[0])
                if CORPUS_START_YEAR <= yr < CORPUS_END_YEAR:
                    decade = (yr // 10) * 10
                    docs_by_decade[decade].append(doc)
            except (ValueError, AttributeError):
                pass
    
    print(f"Documents per decade (1800-1900):")
    total_docs = 0
    for d in sorted(docs_by_decade):
        print(f"  {d}s: {len(docs_by_decade[d])}")
        total_docs += len(docs_by_decade[d])
    print(f"Total: {total_docs}")
    
    # Stratified sampling: allocate chunks proportionally but with minimum for early decades
    # Target: ~5000 chunks per decade minimum for early ones, proportional for later
    target_per_decade = {}
    decades = sorted(docs_by_decade.keys())
    
    # We want at least 5% per decade. With 50k total, that's 2500 per decade.
    # Let's do: min 3000 per decade, rest proportional
    min_per_decade = 3000
    remaining = MAX_CHUNKS - min_per_decade * len(decades)
    
    if remaining < 0:
        # If max_chunks too small, just do proportional
        min_per_decade = MAX_CHUNKS // len(decades)
        remaining = 0
    
    # Calculate proportional allocation for remaining
    total_late_docs = sum(len(docs_by_decade[d]) for d in decades if d >= 1830)
    for d in decades:
        if d < 1830:
            target_per_decade[d] = min_per_decade
        else:
            prop = len(docs_by_decade[d]) / total_late_docs if total_late_docs > 0 else 0
            target_per_decade[d] = min_per_decade + int(remaining * prop)
    
    print("\nTarget chunks per decade:")
    for d in decades:
        print(f"  {d}s: {target_per_decade[d]}")
    print(f"Total target: {sum(target_per_decade.values())}")
    
    # Sample documents per decade
    import random
    random.seed(42)
    selected_docs = []
    for d in decades:
        docs = docs_by_decade[d]
        target = target_per_decade[d]
        # Each doc yields ~1-2 chunks on average, so sample 2x docs
        sample_size = min(len(docs), target * 2)
        sampled = random.sample(docs, sample_size)
        selected_docs.extend(sampled)
        print(f"  {d}s: sampled {len(sampled)} docs (target {target} chunks)")
    
    # Process all selected documents
    all_chunks = []
    for doc in selected_docs:
        all_chunks.extend(process_document(doc))
        if len(all_chunks) >= MAX_CHUNKS:
            break
    
    # Limit to max_chunks
    all_chunks = all_chunks[:MAX_CHUNKS]
    
    # Report actual decade distribution
    actual_decades = defaultdict(int)
    for c in all_chunks:
        yr = c.get('publication_year')
        if yr:
            actual_decades[(yr // 10) * 10] += 1
    
    print(f"\nActual chunks per decade (after processing):")
    for d in sorted(actual_decades):
        pct = actual_decades[d] / len(all_chunks) * 100
        print(f"  {d}s: {actual_decades[d]} ({pct:.1f}%)")
    
    # Check for sparse decades (<5%)
    print("\nSparse decade check (<5% of corpus):")
    for d in sorted(actual_decades):
        pct = actual_decades[d] / len(all_chunks) * 100
        if pct < 5:
            print(f"  WARNING: {d}s only {pct:.1f}%")
    
    # Compute embeddings
    print(f"\nComputing embeddings for {len(all_chunks)} chunks...")
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256)
    for rec, emb in zip(all_chunks, embeddings):
        rec['embedding'] = emb.tolist()
    
    # Save to JSONL
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks_expanded.jsonl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in all_chunks:
            f.write(json.dumps(rec) + '\n')
    print(f"Generated {len(all_chunks)} chunks and saved to {out_path}")
    
    # Also update the main chunks.jsonl for the indexer
    main_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')
    with open(main_path, 'w', encoding='utf-8') as f:
        for rec in all_chunks:
            f.write(json.dumps(rec) + '\n')
    print(f"Also updated {main_path}")

if __name__ == '__main__':
    main()