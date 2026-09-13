import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ranking.ranker import final_search
from explainability.feature_explainer import explain
from retrieval.bm25 import search_bm25
from retrieval.dense import search_dense
from retrieval.hybrid import search_hybrid
from ranking.shared_hybrid import get_hybrid_candidates, clear_hybrid_cache
from ranking.ranker import z_score_normalize
from ranking.normalization_stats import NORM_STATS
from temporal.temporal_parser import TemporalParser
from ranking.temporal_ranker import calculate_temporal_score
from api.canonical_docs import get_canonical, CANONICAL_MAP

app = FastAPI(title="ETHS Retrieval API")

# Allow requests from the React dev server (any localhost port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    size: int = 10

@app.get("/")
def root():
    return {"status": "ETHS API is running"}

@app.post("/search")
def search(req: QueryRequest):
    fetch_size = max(req.size * 3, 30)
    results = final_search(req.query, size=fetch_size)
    clean = []
    seen_text = set()
    for r in results:
        text_key = r.get("text") or ""
        if text_key in seen_text:
            continue
        seen_text.add(text_key)
        clean.append({
            "chunk_id": r.get("chunk_id"),
            "parent_doc_id": r.get("parent_doc_id"),
            "text": r.get("text"),
            "bm25_score": r.get("bm25_score"),
            "dense_score": r.get("dense_score"),
            "temporal_score": r.get("temporal_score"),
            "metadata_score": r.get("metadata_score"),
            "final_score": r.get("final_score"),
            "temporal_explanation": r.get("temporal_explanation"),
            "metadata_explanation": r.get("metadata_explanation"),
            "historical_start": r.get("historical_start"),
            "historical_end": r.get("historical_end"),
        })
        if len(clean) >= req.size:
            break
    return {"query": req.query, "results": clean}

@app.post("/explain")
def explain_query(req: QueryRequest):
    fetch_size = max(req.size * 3, 30)
    exp = explain(req.query, size=fetch_size)
    seen_text = set()
    clean = []
    for e in exp:
        text_key = (e.get("snippet") or e.get("text") or "")
        if text_key in seen_text:
            continue
        seen_text.add(text_key)
        clean.append(e)
        if len(clean) >= req.size:
            break
    return {"query": req.query, "explanations": clean}

# Document view endpoint - returns canonicalized full document
@app.get("/document/{parent_doc_id}")
def get_document(parent_doc_id: str):
    """
    Returns the canonical version of a parent document.
    Canonicalization rule: 
    1. Cross-ID: all parent_doc_ids with identical full text map to the same canonical ID
       (the one with the most chunks / longest total text).
    2. Within-ID: among chunks of the canonical document, use the one with longest text.
    
    This ensures reprints (different parent_doc_ids, same text) all resolve to the same
    canonical document page.
    """
    # Resolve to canonical parent_doc_id
    canonical_pid = get_canonical(parent_doc_id)
    
    try:
        from opensearchpy import OpenSearch
        import yaml
        
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        
        INDEX = cfg.get('index_name', 'ethsearch_chunks')
        client = OpenSearch(
            hosts=[{'host': cfg['opensearch']['host'], 'port': cfg['opensearch']['port']}],
            use_ssl=False, verify_certs=False, ssl_show_warn=False
        )
        
        # Search for all chunks with the CANONICAL parent_doc_id
        body = {
            "size": 100,
            "query": {
                "term": {"parent_doc_id": canonical_pid}
            }
        }
        
        resp = client.search(index=INDEX, body=body)
        hits = resp['hits']['hits']
        
        if not hits:
            raise HTTPException(status_code=404, detail=f"Document {canonical_pid} not found")
        
        # Extract all chunks for the canonical document
        chunks = []
        for h in hits:
            source = h['_source']
            chunks.append({
                "chunk_id": source.get("chunk_id"),
                "parent_doc_id": source.get("parent_doc_id"),
                "text": source.get("text", ""),
                "publication_year": source.get("publication_year"),
                "historical_start": source.get("historical_start"),
                "historical_end": source.get("historical_end"),
                "provenance": source.get("provenance", ""),
            })
        
        if not chunks:
            raise HTTPException(status_code=404, detail=f"Document {canonical_pid} not found")
        
        # Within-ID canonicalization: pick the chunk with the longest text
        canonical_chunk = max(chunks, key=lambda c: len(c["text"]))
        
        # Build full document text by concatenating all chunks in chunk_id order
        chunks_sorted = sorted(chunks, key=lambda c: c["chunk_id"])
        full_text = "\n\n".join(c["text"] for c in chunks_sorted)
        
        # Use the canonical chunk's metadata for the document
        doc_metadata = {
            "parent_doc_id": canonical_chunk["parent_doc_id"],
            "publication_year": canonical_chunk["publication_year"],
            "historical_start": canonical_chunk["historical_start"],
            "historical_end": canonical_chunk["historical_end"],
            "provenance": canonical_chunk["provenance"],
            "num_chunks": len(chunks),
            "is_canonical": parent_doc_id == canonical_pid,
            "original_pid": parent_doc_id,
        }
        
        return {
            "parent_doc_id": canonical_pid,
            "full_text": full_text,
            "chunks": chunks_sorted,
            "metadata": doc_metadata,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.get("/canonical-map/stats")
def canonical_stats():
    """Return statistics about the canonical mapping."""
    groups = {}
    for pid, canonical in CANONICAL_MAP.items():
        if canonical not in groups:
            groups[canonical] = []
        groups[canonical].append(pid)
    
    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    total_duplicates = sum(len(v) - 1 for v in duplicate_groups.values())
    
    return {
        "total_parent_doc_ids": len(CANONICAL_MAP),
        "unique_canonical_ids": len(groups),
        "duplicate_groups": len(duplicate_groups),
        "total_duplicate_ids": total_duplicates,
        "example_groups": {k: v for i, (k, v) in enumerate(duplicate_groups.items()) if i < 5}
    }