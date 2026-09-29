import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# python-dotenv is a dependency, but nothing loaded it, so a .env file was
# silently ignored and the overview key never reached the app. This must run
# BEFORE api.overview is imported, because that module reads its configuration
# from the environment at import time.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
from api.loc_images import document_references, resolve_page_image
from api.autocomplete import suggest as autocomplete_suggest, index_meta as autocomplete_meta
from api.corpus_stats import decade_density as corpus_density
from api.overview import generate_overview

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
    # Optional hard year constraint, applied during retrieval rather than as a
    # post-hoc filter over the results already returned.
    year_start: int | None = None
    year_end: int | None = None

class OverviewRequest(BaseModel):
    query: str
    size: int = 10
    max_sources: int = 6
    year_start: int | None = None
    year_end: int | None = None

def _dedupe_by_text(results, limit):
    """Keep the first result for each distinct passage text, up to `limit`.

    Shared by /search and /overview on purpose. The overview assigns source
    numbers by position, so without this two copies of the same passage would
    become two independently-cited "sources" and make a single piece of evidence
    look like corroboration.
    """
    clean = []
    seen_text = set()
    for r in results:
        text_key = r.get("text") or ""
        if text_key in seen_text:
            continue
        seen_text.add(text_key)
        clean.append(r)
        if len(clean) >= limit:
            break
    return clean

@app.get("/")
def root():
    return {"status": "ETHS API is running"}

@app.post("/search")
def search(req: QueryRequest):
    fetch_size = max(req.size * 3, 30)
    results = final_search(
        req.query, size=fetch_size,
        year_start=req.year_start, year_end=req.year_end,
    )
    clean = [{
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
        "publication_year": r.get("publication_year"),
        "historical_start": r.get("historical_start"),
        "historical_end": r.get("historical_end"),
    } for r in _dedupe_by_text(results, req.size)]
    applied = None
    if req.year_start is not None or req.year_end is not None:
        applied = {"year_start": req.year_start, "year_end": req.year_end}
    return {"query": req.query, "results": clean, "year_range": applied}

@app.post("/overview")
def overview(req: OverviewRequest):
    """Grounded AI overview of the retrieved results for a query.

    The request carries only a query string. The snippets summarised here are
    re-retrieved server-side from that query, so a client cannot smuggle in
    context of its own and have it appear as a grounded, cited summary.

    Note this is not the same as /search's top results: it re-runs retrieval
    rather than reusing the search response, which is the deliberate cost of
    keeping the model's input trustworthy. (The hybrid candidate cache makes the
    second retrieval a cache hit, so this is not a second dense pass.)
    """
    if not (req.query or "").strip():
        raise HTTPException(status_code=400, detail="query is required")
    # The same year range the results list is showing is applied here, so the
    # summary always describes the passages the user can actually see. A summary
    # of the unfiltered corpus beside a filtered result list would be describing
    # a different set of evidence than the one on screen.
    results = final_search(
        req.query, size=max(req.size, 20),
        year_start=req.year_start, year_end=req.year_end,
    )
    results = _dedupe_by_text(results, max(req.size, 20))
    return generate_overview(
        req.query, results,
        max_sources=max(1, min(req.max_sources, 12)),
    )

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
            # Library of Congress page-image reference, derived from the
            # chunk provenance URLs. Absent when no chunk carries a
            # parseable Chronicling America page URL.
            "source_image": document_references(c.get("provenance") for c in chunks_sorted),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")


@app.get("/page-image")
def get_page_image(lccn: str, date: str, edition: int, sequence: int):
    """Resolve a Chronicling America page to a browser-displayable JPEG.

    The corpus only yields JP2/PDF URLs, and JP2 is not renderable by browsers,
    so display requires LOC's IIIF image service. Those identifiers live in the
    issue manifest, which is resolved here rather than in the browser: this
    keeps the request same-origin (no CORS) and lets the manifest be fetched from
    the Internet Archive when loc.gov's WAF refuses this network.

    Resolution is cached per issue. Every failure returns HTTP 200 with
    available=false and a reason, so the client can degrade to plain LOC links
    rather than treating absence as an error.
    """
    return resolve_page_image(lccn, date, edition, sequence)


@app.get("/autocomplete")
def autocomplete(q: str = "", limit: int = 8):
    """Prefix-matched query suggestions from the corpus vocabulary.

    Suggestions come only from terms, entities and years that occur in the
    indexed documents, so autocomplete never proposes a query the corpus cannot
    answer. Returns an empty list when the prefix is too short or the index has
    not been built; the frontend treats that as "no dropdown", not an error.
    """
    return {
        "prefix": q,
        "suggestions": autocomplete_suggest(q, limit),
    }


@app.get("/autocomplete/meta")
def autocomplete_index_info():
    """Report what the suggestion index was built from, for verification."""
    return autocomplete_meta()


@app.get("/corpus/decade-density")
def corpus_decade_density(corpus: str = "expanded"):
    """Per-decade document density for the timeline's background layer.

    Served from a precomputed artifact rather than aggregated live, so the
    context band costs the same regardless of corpus size. The returned range
    reflects the years actually present, which is not the same as the
    configured corpus_end_year: the indexed data stops short of it.

    Returns 503 rather than an empty payload when the artifact has not been
    built, so the client can distinguish "not built" from "no documents".
    """
    payload = corpus_density(corpus)
    if payload is None:
        raise HTTPException(
            status_code=503,
            detail="decade density artifact not built; run evaluation/build_decade_density.py",
        )
    return payload


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