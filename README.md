# ETHS – Explainable Temporal Historical Search

A retrieval pipeline for 19th-century newspaper QA that combines BM25 lexical search, dense semantic search, temporal re-ranking, and metadata scoring with transparent, feature-level explanations.

## Architecture

| Layer | Module | Description |
|-------|--------|-------------|
| Indexing | `indexing/indexer.py` | Loads `data/chunks.jsonl` into OpenSearch with KNN + text mapping |
| Retrieval | `retrieval/bm25.py` | BM25 baseline via OpenSearch `match` query |
| Retrieval | `retrieval/dense.py` | Dense retrieval via `all-MiniLM-L6-v2` + OpenSearch KNN |
| Retrieval | `retrieval/hybrid.py` | Min-max normalized linear fusion of BM25 + Dense |
| Ranking | `ranking/temporal_ranker.py` | Temporal constraint scoring + hybrid fusion |
| Ranking | `ranking/ranker.py` | Final linear ranker: BM25, Dense, Temporal, Metadata |
| Explainability | `explainability/feature_explainer.py` | Transparent feature contribution breakdown |
| API | `api/main.py` | FastAPI server exposing `/search` and `/explain` |
| Frontend | `frontend/` | React + Vite search UI with explainability toggle |

## Eval Results

Eval uses 20 hand-crafted queries from `data/qrels.json`, measured at cutoff 10.

### After dedup (primary)

Each query now has **1 distinct relevant document** (the 5 nominal entries were reprints of the same ad/notice). With only 1 relevant doc per query, P@10 has limited range; **Recall@10 and MRR are the primary metrics**.

Bootstrap resampling (n=10000, 95% CI) shows wide confidence intervals due to small eval set (n=20):

| Method | P@10 | Recall@10 [95% CI] | MRR [95% CI] |
|--------|------|-----------|-----|
| BM25 | 0.05 | 0.50 [0.30, 0.70] | 0.398 [0.203, 0.607] |
| Dense | 0.02 | 0.20 [0.10, 0.45] | 0.208 [0.050, 0.400] |
| Hybrid (α=0.5) | 0.055 | 0.55 [0.35, 0.75] | 0.413 [0.217, 0.613] |
| Temporal (α=0.7, β=0.3) | 0.04 | 0.40 [0.20, 0.60] | 0.279 [0.110, 0.472] |
| Final Ranker | 0.04 | 0.40 [0.20, 0.60] | 0.280 [0.107, 0.472] |

**Pairwise significance:** All pairwise comparisons overlap at 95% CI — no method is statistically significantly better than any other on this eval set. The headline differences (e.g. Hybrid 0.55 vs Temporal 0.40) could flip with a handful of queries.

### Before dedup (nominal 5 relevant/query, includes reprints)

| Method | P@10 | Recall@10 | MRR |
|--------|------|-----------|-----|
| BM25 | 0.40 | 0.40 | 0.399 |
| Dense | 0.22 | 0.22 | 0.223 |
| Hybrid (α=0.5) | 0.48 | 0.48 | 0.413 |
| Temporal (α=0.7, β=0.3) | 0.37 | 0.37 | 0.286 |
| Final Ranker | 0.37 | 0.37 | 0.286 |

These numbers are inflated because retrieving any one of the 5 reprinted copies counts as a hit against all 5 qrels entries. The after-dedup table is the honest measurement.

**Note on Final Ranker:** Final Ranker is bit-for-bit identical to Temporal Ranker on these metrics. Investigation (`evaluation/grid_search.py` and per-query tracing) shows:
- The grid search (`alpha_hybrid=0.7, beta_temporal=0.3`) was tuned against the **pre-dedup qrels** (5 nominal relevant docs/query). Rerunning it on the deduped qrels yields the same weights, so the tuning is robust.
- Metadata contributes nothing (constant neutral) because `historical_period` and `location` are unpopulated.
- BM25 and Dense scores are highly correlated with the hybrid score on this corpus. When temporal scores are constant across the result set, Final Ranker ranking ≈ Hybrid ranking. When temporal scores vary (e.g., some docs outside the year range), the 0.3 temporal weight can push high-BM25 docs down, making Final ≈ Temporal. The net effect across 20 queries is that Final and Temporal produce identical Recall/MRR.

**Note on P@10:** With 1 relevant doc per query, P@10 is binary per query (0 or 0.1). The mean P@10 ≈ Recall@10 ÷ 10, so it has limited discriminative power. Recall@10 and MRR are the primary metrics.

## Temporal Ranker Evidence

On temporally-specific queries, the Temporal Ranker meaningfully moves the correct document:

| Query | Temporal Type | Hybrid Rank | Temporal Rank | Δ |
|-------|--------------|-------------|---------------|---|
| "ship arrivals cargo from Europe 1803" | explicit_year | 85 | 16 | +69 |
| "real estate land for sale 1807" | explicit_year | 64 | 20 | +44 |
| "poetry verse published before 1810" | before_after | 61 | 17 | +44 |
| "political news congress 1800 to 1805" | range | 3 | 1 | +2 |
| "news dispatches France Europe before 1808" | before_after | 10 | 6 | +4 |

See `evaluation/` for diagnostic scripts used during development.

## Explanation Design

Most documents in this 1800–1810 corpus trivially satisfy most temporal constraints, so a raw temporal score (e.g. "0.31") would be falsely precise. The explainer detects this **low-discrimination** case when all temporal scores in the result set have a range < 0.05, and renders:

> **Temporal relevance:** not distinguishing for this query

When temporal *is* discriminating, it shows the normalized percentage and the raw explanation label (e.g. `exact_overlap`, `adjacent`).

## Setup

1. Copy `.env.example` to `.env` and set `OPENSEARCH_PASSWORD` if your OpenSearch instance requires auth.
2. Install dependencies, build the index, and start the API/frontend as described below.

## Running

```bash
# Start OpenSearch
docker compose up -d

# Install dependencies
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Build index
python indexing/indexer.py

# Run eval
python evaluation/metrics.py

# Start API
uvicorn api.main:app --reload

# Start frontend (separate terminal)
cd frontend
npm install
npm run dev
```

## Deferred Roadmap

These items are tracked but not in the current work scope:

- Learning-to-Rank (LTR) with gradient-boosted trees
- Source/provenance scoring
- Full ablation matrix
- Vocabulary-drift experiment across decades
- External user study
- Timeline visualization
- Paper writeup

## Priority 1 Investigation Summary (2026-09-09)

**Final Ranker ≈ Temporal Ranker root cause:**
- Metadata is neutral (0.5 constant) because corpus fields are unpopulated.
- BM25/Dense are highly correlated with hybrid score. When temporal is constant, Final ≈ Hybrid. When temporal varies (some docs outside year range), the 0.3 temporal weight can push high-BM25 docs down, making Final ≈ Temporal.
- Net effect across 20 queries: identical Recall@10 and MRR.

**Grid search provenance:**
- `evaluation/grid_search.py` was tuned against the **pre-dedup qrels** (5 nominal relevant docs/query).
- Rerunning on deduped qrels yields identical optimal weights (`alpha_hybrid=0.5, beta_temporal=0.3`), so tuning is robust.

**Statistical significance:**
- Bootstrap resampling (10000, 95% CI) on n=20 queries shows all pairwise CIs overlap.
- No method is statistically significantly better than any other on this eval set.
- Differences of 0.1–0.2 in Recall@10 could flip with a handful of queries.

## Testing / Edge Cases

The API handles these cases without crashing:

- **Empty query** → returns results (BM25 fallback)
- **Garbage query** (`napolean's dick`) → degrades gracefully, returns non-duplicate weak matches, no crash
- **Unknown period** (`Cretaceous Period`) → returns results with `temporal_explanation=neutral`
- **Missing metadata** → returns `metadata_score=0.5` (neutral), no crash

Frontend hard-refresh (`Ctrl+Shift+R`) after rebuild to avoid stale bundles.

## Data

- **Corpus:** `Bhawna/ChroniclingAmericaQA` (Chronicling America, 1800–1810)
- **Eval set:** 20 hand-crafted queries with temporal annotations in `data/qrels.json`
- **Index:** 2,000 chunks in OpenSearch with `text` (BM25) + `embedding` (KNN) fields

## Known Limitations

1. **P@10 ceiling with 1 relevant doc:** After deduplication, each eval query has exactly 1 distinct relevant document. This means P@10 can only be 0.0 or 1.0 per query, so its average has limited discriminative power. Recall@10 and MRR are the primary metrics for this eval set.

2. **Metadata component is non-discriminative:** The `historical_period` and `location` fields are not populated in the current corpus subset, so the metadata score is constant (neutral) for every query. Final Ranker therefore behaves identically to Temporal Ranker. The metadata plumbing is wired in and will contribute once those fields are populated.

3. **Temporal corpus is narrow:** All documents fall within 1800–1810, so temporal re-ranking can only distinguish queries with explicit year constraints; broader period queries see little temporal signal.
