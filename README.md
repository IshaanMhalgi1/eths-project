# ETHS – Explainable Temporal Historical Search

A retrieval pipeline for 19th-century newspaper QA that combines BM25 lexical search, dense semantic search, temporal re-ranking, and metadata scoring with transparent, feature-level explanations.

## Architecture

| Layer | Module | Description |
|-------|--------|-------------|
| Indexing | `indexing/indexer.py` | Loads `data/chunks.jsonl` into OpenSearch with KNN + text mapping |
| Retrieval | `retrieval/bm25.py` | BM25 baseline via OpenSearch `match` query |
| Retrieval | `retrieval/dense.py` | Dense retrieval via `all-MiniLM-L6-v2` + OpenSearch KNN |
| Retrieval | `retrieval/hybrid.py` | RRF rank-fusion of BM25 + Dense (fetch-size invariant) |
| Ranking | `ranking/shared_hybrid.py` | Single shared hybrid candidate pool (fetch_size=50) for all rankers |
| Ranking | `ranking/temporal_ranker.py` | Temporal constraint scoring + hybrid fusion (α=0.7, β=0.3) |
| Ranking | `ranking/ranker.py` | Final linear ranker: Hybrid (70%) + Temporal (20%) + Metadata (10%) |
| Explainability | `explainability/feature_explainer.py` | Transparent feature contribution breakdown |
| API | `api/main.py` | FastAPI server exposing `/search` and `/explain` |
| Frontend | `frontend/` | React + Vite search UI with explainability toggle |

## Eval Results (standardized, fetch_size=50)

Eval uses 20 hand-crafted queries from `data/qrels.json`, measured at cutoff 10. All rankers use a shared hybrid candidate pool with fetch_size=50 (matching the pipeline's `size * 5` design).

### After dedup (primary)

Each query now has **1 distinct relevant document** (the 5 nominal entries were reprints of the same ad/notice). With only 1 relevant doc per query, P@10 has limited range; **Recall@10 and MRR are the primary metrics**.

Bootstrap resampling (n=10000, 95% CI) shows wide confidence intervals due to small eval set (n=20):

| Method | P@10 | Recall@10 [95% CI] | MRR [95% CI] |
|--------|------|-----------|-----|
| BM25 | 0.05 | 0.50 [0.30, 0.70] | 0.399 [0.207, 0.607] |
| Dense | 0.025 | 0.25 [0.10, 0.45] | 0.208 [0.050, 0.400] |
| Hybrid (α=0.5) | 0.040 | 0.40 [0.20, 0.60] | 0.248 [0.089, 0.432] |
| Temporal (α=0.7, β=0.3) | 0.040 | 0.40 [0.20, 0.60] | 0.284 [0.110, 0.480] |
| Final Ranker | 0.040 | 0.40 [0.20, 0.60] | 0.281 [0.110, 0.475] |

**Pairwise significance:** All pairwise comparisons overlap at 95% CI — no method is statistically significantly better than any other on this eval set.

### Before dedup (nominal 5 relevant/query, includes reprints)

| Method | P@10 | Recall@10 | MRR |
|--------|------|-----------|-----|
| BM25 | 0.40 | 0.40 | 0.399 |
| Dense | 0.22 | 0.22 | 0.223 |
| Hybrid (α=0.5) | 0.48 | 0.48 | 0.413 |
| Temporal (α=0.7, β=0.3) | 0.37 | 0.37 | 0.286 |
| Final Ranker | 0.37 | 0.37 | 0.286 |

These numbers are inflated because retrieving any one of the 5 reprinted copies counts as a hit against all 5 qrels entries. The after-dedup table is the honest measurement.

**Note on Final Ranker:** Final Ranker and Temporal Ranker produce identical Recall/MRR on the current eval set because they share the same hybrid base (70% weight) and same temporal adjustment — they differ only in temporal weight (0.2 vs 0.3) and metadata (0.1, neutral). This is a design consequence, not a bug.

**Critical finding — temporal re-ranking is net-negative:** Across three independent debugging passes (original ranker, normalization fix, wiring fix), including temporal reranking at any material weight consistently drops MRR relative to pure Hybrid (0.412 → ~0.28–0.29). This is a replicated result, not an artifact of the bugs that were fixed along the way, and suggests the current temporal reranking approach may be net-negative for this task on this eval set.

**Grid search provenance:** The grid search (`evaluation/grid_search.py`) tested `beta_temporal ∈ {0.1, 0.2, 0.3, 0.4, 0.5}` but **did not include β=0 (temporal off)** as a candidate. The reported "optimal" β=0.3 was only the best among nonzero temporal weights. When β=0 is added to the search space with standardized fetch_size=50, pure Hybrid (β=0) MRR=0.249 is exceeded only by β=0.4–0.5 (MRR=0.290). The previously reported optimum was only the best among configurations that included temporal weight.

The **wiring bug** (each ranker independently fetching hybrid candidates with `size * 5`, causing different candidate pools and inconsistent RRF scores) was fixed in `ranking/shared_hybrid.py`. All rankers now draw from a single shared hybrid candidate pool (fixed size 50, matching the pipeline's `size * 5` design), ensuring consistent scores and rankings.

- Metadata contributes nothing (constant neutral) because `historical_period` and `location` are unpopulated.
- Temporal re-ranking helps on constrained queries (e.g., "ship arrivals 1803": Hybrid rank 85 → Temporal rank 16), but net effect across 20 queries is small.

**Finding — Hybrid (RRF) underperforms BM25 at fetch_size=50 with current params (k=60, alpha=0.5):** When RRF fetch_size is standardized to 50 (the pipeline's design intent), Hybrid MRR=0.249 vs BM25 MRR=0.399. Root cause: RRF k=60 compresses scores to range [0.0164, 0.0091] at fetch_size=50; equal weighting (alpha=0.5) dilutes BM25's strong signal with weak Dense. **This finding is specific to the current RRF parameters (k=60, alpha=0.5) — no sweep over k or alpha has been run.** The previously reported Hybrid MRR=0.412 was measured at unstable fetch_size=20 and is not reproducible at the design fetch_size.

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

4. **Eval qrels lexical bias (applies to both corpora):** The qrels for both the 1800–1810 and 1800–1900 corpora were built using independent term-overlap matching (`generate_independent_qrels.py`, `generate_expanded_qrels.py`), NOT retriever outputs. While this avoids circularity (retriever → qrels → eval), term-overlap favors BM25 lexically — queries and relevant docs share surface words by construction. This is the same bias flagged as bug #6 on the original corpus. BM25's advantage over Hybrid/Dense may be partially inflated by this evaluation design. A non-lexical qrels method (e.g., human annotation or LLM-based relevance) would be needed to fully isolate retriever quality from lexical matching.

## Priority 2: Corpus Expansion to 1800–1900 (2026-09-12)

Expanded corpus from 10-year (1800–1810, ~2,000 chunks) to full 19th century (1800–1900, 50,000 chunks) using stratified decade sampling (all decades ≥5% of corpus). Re-ran full eval pipeline with 90 queries (75–100 target) built via independent term-overlap matching (no retriever circularity).

### Expanded Corpus Stats
- **Chunks:** 50,000 (vs 2,000)
- **Decade distribution:** 1800s 7%, 1810s 8%, 1820s 12%, 1830s 17%, 1840s 20%, 1850s 23%, 1860s 13%, 1870s–1890s sparse (<5% each)
- **Deduplication:** 0 chunk-ID dupes, 25 parent-doc-ID dupes (same-year), 11,416 full-text dupes (all same-decade, no cross-decade reprints found)

### Eval Results (90 queries, fetch_size=50, shared pool, global z-score)

| Method | P@10 | Recall@10 [95% CI] | MRR [95% CI] |
|--------|------|-----------|-----|
| BM25 | 0.131 | 0.271 [0.212, 0.334] | **0.322** [0.248, 0.399] |
| Dense | 0.029 | 0.058 [0.033, 0.087] | 0.107 [0.057, 0.162] |
| Hybrid (α=0.5) | 0.058 | 0.124 [0.082, 0.167] | 0.167 [0.109, 0.231] |
| Temporal (α=0.7, β=0.3) | 0.095 | 0.198 [0.149, 0.249] | 0.271 [0.200, 0.348] |
| Final Ranker | 0.085 | 0.176 [0.131, 0.225] | 0.253 [0.182, 0.327] |

**Pairwise significance (Recall@10, bootstrap n=10000, 95% CI):**
- BM25 > Dense: **significant** (diff=+0.213, CI=[+0.142, +0.284])
- BM25 > Hybrid (α=0.5): **significant** (diff=+0.120, CI=[+0.067, +0.176])
- Temporal (β=0.5) > Hybrid (β=0): **significant** (diff=+0.118, CI=[+0.076, +0.167])
- Temporal (β=0.5) > Dense: **significant** (diff=+0.140, CI=[+0.081, +0.201])
- BM25 vs Temporal (β=0.5): **not significant** (diff=+0.002, CI=[-0.058, +0.064])
- Temporal vs Final, BM25 vs Final: CIs overlap (not significant)

### Critical Findings on Expanded Corpus

1. **BM25 dominates:** Pure BM25 (MRR=0.322) outperforms all Hybrid RRF configurations. RRF parameter sweep (k∈{10,20,40,60,80,100}, α∈[0,1]) confirms **no Hybrid configuration beats BM25** — best Hybrid = pure BM25 (α=1.0). Dense alone MRR=0.130.

2. **Temporal reranking is STRONGLY POSITIVE** (opposite of 1800–1810 finding): Grid search (β∈{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}) on expanded corpus + 90 queries: optimal β=0.5 (MRR=0.387, Recall@10=0.269) vs β=0 (MRR=0.204). MRR plateaus at β=0.5 and stays flat through β=1.0 — optimum is not at grid edge. Temporal signal is now discriminative across 100 years of content. **Bootstrap confirms temporal reranking benefit is significant** (β=0.5 vs β=0: diff=+0.118 Recall@10, +0.174 MRR, both p<0.05).

3. **All three eval paths agree on pure-Hybrid baseline:** 
   - `metrics.py` (shared pool): Hybrid MRR=0.167
   - `grid_search.py` β=0 (shared pool): Hybrid MRR=0.204
   - `bootstrap_significance.py` Hybrid row: MRR=0.167
   - Minor discrepancy between grid search and metrics/bootstrap under investigation (different temporal scoring paths), but all agree Hybrid < BM25.

### Content Coverage Check
- **Native American content:** Substantial — Cherokee (200), Sioux (85), Apache (25), Navajo (13), Comanche (16), Iroquois (26), Creek (1,026), Seminole (82), Choctaw (55), Chickasaw (38), tribe (377), reservation (248), treaty (983), Custer (9), Indian Territory (10).
- **India/South Asia content:** Present — British India (5), East India Company (23), Calcutta (102), Bombay (44), Madras (86), Delhi (19), Sepoy (2), Mughal (1), Bengal (62), Punjab (3), Ceylon (25). Term "indian" (2,766) is ambiguous (Native American vs. India).

### Updated Architecture Notes
- **Ranker weights re-tuned on expanded corpus:** Grid search optimal β=0.5 (temporal weight), suggesting `alpha_hybrid=0.5, beta_temporal=0.5` for Temporal Ranker. Final Ranker weights should be revisited.
- **RRF parameters:** No k/α combination restored Hybrid ≥ BM25. Consider dropping RRF fusion or using BM25-only for this corpus.
- **Eval stability:** 90 queries narrowed CIs enough for significant pairwise results (BM25 > Hybrid, Temporal > Hybrid(β=0), BM25 > Dense, Temporal > Dense).
