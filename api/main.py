import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ranking.ranker import final_search
from explainability.feature_explainer import explain

app = FastAPI(title="ETHS Retrieval API")

# Allow requests from the React dev server (any localhost port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for development – restrict in production
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
    # Fetch extra candidates so we can deduplicate by text while still returning `size` distinct results
    fetch_size = max(req.size * 3, 30)
    results = final_search(req.query, size=fetch_size)
    # Make results JSON-serialisable (remove non-serialisable fields if any)
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
    # Fetch extra candidates so we can deduplicate by text while still returning `size` distinct results
    fetch_size = max(req.size * 3, 30)
    exp = explain(req.query, size=fetch_size)
    # Deduplicate by text content for user-facing output
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
