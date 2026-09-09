import os
import json
import uuid
from typing import List, Dict

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
import yaml
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

EMBEDDING_MODEL = cfg.get('embedding_model', 'all-MiniLM-L6-v2')
MAX_CHUNKS = cfg.get('max_chunks', 2000)

model = SentenceTransformer(EMBEDDING_MODEL)

def clean_text(text: str) -> str:
    # Basic cleaning: strip, replace multiple spaces, remove non‑printable chars
    return " ".join(text.split())

def chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    # Very simple chunker: split by sentence punctuation, accumulate until token limit
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    cur_len = 0
    for sent in sentences:
        # Approx token count using whitespace split (good enough for MVP)
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
    # ChroniclingAmericaQA keys: 'query_id', 'question', 'answer', 'org_answer', 'para_id', 'context', 'raw_ocr', 'publication_date', 'trans_que', 'trans_ans', 'url'
    parent_id = str(doc.get('query_id', uuid.uuid4()))
    raw_text = clean_text(doc.get('context') or '')
    
    # Extract year from publication_date (e.g. "1905-08-20")
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
            "title": None,  # Not in dataset
            "author": None, # Not in dataset
            "text": chunk,
            "publication_year": pub_year,
            "historical_start": pub_year,
            "historical_end": pub_year,
            "location": None, # Not in dataset
            "historical_period": None, # Not in dataset
            "topic": None, # Not in dataset
            "document_type": None, # Not in dataset
            "source": "Chronicling America",
            "provenance": doc.get('url', ''),
        }
        records.append(record)
    return records

def main():
    dataset = load_dataset('Bhawna/ChroniclingAmericaQA', split='train')
    # Take a small subset
    subset = dataset.select(range(min(MAX_CHUNKS, len(dataset))))
    all_chunks = []
    for doc in subset:
        all_chunks.extend(process_document(doc))
        if len(all_chunks) >= MAX_CHUNKS:
            break
    # Limit to max_chunks
    all_chunks = all_chunks[:MAX_CHUNKS]
    # Compute embeddings
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    for rec, emb in zip(all_chunks, embeddings):
        rec['embedding'] = emb.tolist()
    # Save to JSONL for later bulk indexing
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.jsonl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in all_chunks:
            f.write(json.dumps(rec) + '\n')
    print(f"Generated {len(all_chunks)} chunks and saved to {out_path}")

if __name__ == '__main__':
    main()
