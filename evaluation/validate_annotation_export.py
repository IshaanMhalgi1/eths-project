"""
Validate an annotator export against the blind v2 annotation pool.

Checks structural integrity, pool membership, and blind-protocol compliance
before the file is allowed anywhere near the analysis pipeline.

Usage:
    python evaluate/validate_annotation_export.py \
        --export C:/Users/.../annotation_results_annotator1_1790430041333.json \
        --pool data/annotation/annotation_pools.json \
        --annotator 1
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--pool", default=str(REPO / "data" / "annotation" / "annotation_pools.json"))
    ap.add_argument("--annotator", type=int, required=True, choices=[1, 2])
    args = ap.parse_args()

    export = load_json(Path(args.export))
    pool = load_json(Path(args.pool))

    pool_by_qid = {q["query_id"]: q for q in pool["queries"]}
    a = args.annotator
    rel_key = f"relevance_annotator{a}"
    notes_key = f"notes_annotator{a}"

    errors, warnings = [], []

    print("=" * 74)
    print("ANNOTATION EXPORT VALIDATION")
    print("=" * 74)
    print(f"export    : {Path(args.export).name}")
    print(f"version   : {export.get('version')}")
    print(f"annotator : {export.get('annotator')} (expected {a})")
    print(f"exported  : {export.get('export_date')}")
    print(f"queries   : {len(export.get('queries', []))}")
    print()

    if export.get("annotator") != a:
        errors.append(
            f"annotator field is {export.get('annotator')!r}, expected {a}. "
            "Refusing to guess which side produced this file."
        )

    # ---- blind protocol: no retriever identity anywhere in the file ----
    raw = json.dumps(export).lower()
    for banned in ["retriever_scores", "bm25", "dense", "hybrid", "raw_score", "rrf"]:
        if banned in raw:
            errors.append(f"blind-protocol leak: export contains {banned!r}")

    # ---- per-query checks ----
    print("-" * 74)
    print(f"{'query':<8} {'in_pool':<8} {'cands':>6} {'pool':>5} "
          f"{'labeled':>8} {'rel(0)':>7} {'rel(1)':>7} {'rel(2)':>7}  status")
    print("-" * 74)

    total_labeled = 0
    total_pool = 0
    grade_counts = {0: 0, 1: 0, 2: 0}
    seen_qids = set()

    for eq in export.get("queries", []):
        qid = eq.get("query_id")
        seen_qids.add(qid)
        cands = eq.get("candidates", [])
        n = len(cands)

        if qid not in pool_by_qid:
            errors.append(f"{qid}: not present in the v2 pool")
            print(f"{qid:<8} {'NO':<8} {n:>6} {'-':>5} {'-':>8} "
                  f"{'-':>7} {'-':>7} {'-':>7}  UNKNOWN QUERY")
            continue

        pool_cands = pool_by_qid[qid]["candidates"]
        pool_ids = {c["canonical_parent_doc_id"] for c in pool_cands}
        pool_n = len(pool_cands)

        # candidate_id alignment against the pool
        export_ids = {c.get("canonical_parent_doc_id") for c in cands}
        missing = pool_ids - export_ids
        extra = export_ids - pool_ids
        if missing:
            errors.append(f"{qid}: {len(missing)} pool candidates absent from export")
        if extra:
            errors.append(f"{qid}: {len(extra)} export candidates not in pool: {sorted(extra)}")
        if n != pool_n:
            warnings.append(f"{qid}: {n} candidates in export vs {pool_n} in pool")

        labeled, g = 0, {0: 0, 1: 0, 2: 0}
        for c in cands:
            v = c.get(rel_key)
            if v is not None:
                labeled += 1
                if v in (0, 1, 2):
                    g[v] += 1
                else:
                    errors.append(f"{qid}: invalid {rel_key}={v!r} on {c.get('canonical_parent_doc_id')}")

        for k in (0, 1, 2):
            grade_counts[k] += g[k]
        total_labeled += labeled
        total_pool += pool_n

        complete = (labeled == pool_n) and not missing and not extra
        status = "complete" if complete else f"PARTIAL {labeled}/{pool_n}"
        if not complete and labeled == 0:
            status = "UNLABELED"
            errors.append(f"{qid}: no labels at all")
        elif not complete:
            warnings.append(f"{qid}: only {labeled}/{pool_n} candidates labeled")

        print(f"{qid:<8} {'yes':<8} {n:>6} {pool_n:>5} {labeled:>8} "
              f"{g[0]:>7} {g[1]:>7} {g[2]:>7}  {status}")

    # ---- coverage of the whole pool ----
    print("-" * 74)
    missing_q = set(pool_by_qid) - seen_qids
    if missing_q:
        warnings.append(f"queries absent from export: {sorted(missing_q)}")

    print()
    print(f"total candidates in pool : {total_pool}")
    print(f"total labeled by a{a}       : {total_labeled} "
          f"({100 * total_labeled / total_pool:.1f}%)")
    print(f"grade distribution    : "
          f"0 (not rel)={grade_counts[0]}  1 (partial)={grade_counts[1]}  2 (rel)={grade_counts[2]}")
    if grade_counts[2] == 0:
        errors.append("zero 'Relevant (2)' labels across the whole export")
    if grade_counts[0] == 0:
        warnings.append("zero 'Not Relevant (0)' labels - was the button actually used?")

    print()
    if warnings:
        print(f"WARNINGS ({len(warnings)})")
        for w in warnings:
            print(f"  ! {w}")
        print()
    if errors:
        print(f"ERRORS ({len(errors)}) -> DO NOT MERGE")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)
    print("RESULT: PASS - safe to merge.")
    print("=" * 74)


if __name__ == "__main__":
    main()
