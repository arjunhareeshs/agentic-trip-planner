"""
run_eval.py — RAG Evaluation Runner.

Ingests PDFs, runs ground-truth queries, computes IR metrics,
and writes detailed results to the output/ folder.

NOT meant to be run directly — use test_metrics.py as the entry point:

    python -m test.ragtest.test_metrics --pdf "C:/path/to/your.pdf"
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure project root is on sys.path ───────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent           # test/ragtest/
_SRC_DIR = _SCRIPT_DIR.parent.parent                    # server/src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from RAG import RAGPipeline, RAGResult, RAGContext
from test.ragtest.metrics import (
    compute_all_metrics,
    average_metrics,
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)

OUTPUT_DIR = _SCRIPT_DIR / "output"


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _load_dataset(path: str) -> Dict[str, Any]:
    """Load and validate the JSON test dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")
    if p.suffix.lower() != ".json":
        raise ValueError(f"Expected .json file, got: {p.suffix}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "queries" not in data or not data["queries"]:
        raise ValueError("Dataset must contain a non-empty 'queries' array.")

    return data


def _keyword_match(content: str, keywords: List[str]) -> bool:
    """Check if content contains ANY of the keywords (case-insensitive)."""
    content_lower = content.lower()
    return any(kw.lower() in content_lower for kw in keywords)


def _extract_retrieved_ids(
    context: RAGContext,
    relevant_keywords: Optional[List[str]] = None,
) -> tuple:
    """
    Extract ordered list of retrieved node IDs from RAGContext.
    Also returns a mapping of node_id → content for keyword matching.

    Returns:
        (retrieved_ids, id_to_content_map)
    """
    retrieved_ids: List[str] = []
    id_to_content: Dict[str, str] = {}

    # Text nodes come first (higher priority after reranking)
    for node in context.retrieved_text_nodes:
        retrieved_ids.append(node.node_id)
        id_to_content[node.node_id] = node.content

    # Then image nodes
    for node in context.retrieved_image_nodes:
        retrieved_ids.append(node.node_id)
        id_to_content[node.node_id] = node.content

    return retrieved_ids, id_to_content


def _resolve_relevant_ids(
    retrieved_ids: List[str],
    id_to_content: Dict[str, str],
    ground_truth_ids: Optional[List[str]],
    keywords: Optional[List[str]],
) -> List[str]:
    """
    Determine which node IDs are relevant.

    Strategy:
      1. If ground_truth_ids is provided and non-empty → use those directly.
      2. Else if keywords given → scan retrieved content for keyword matches
         and treat matching nodes as relevant.
      3. Else → return empty (all metrics will be 0).
    """
    if ground_truth_ids:
        return ground_truth_ids

    if keywords:
        # Keyword-based relevance: any retrieved node containing a keyword is relevant
        return [
            nid for nid in retrieved_ids
            if _keyword_match(id_to_content.get(nid, ""), keywords)
        ]

    return []


def _format_summary_table(
    per_query: List[Dict[str, Any]],
    aggregate: Dict[str, float],
    k_values: List[int],
    elapsed: float,
) -> str:
    """Build a human-readable summary text."""
    lines = [
        "=" * 70,
        "  RAG EVALUATION REPORT",
        f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Queries:  {len(per_query)}",
        f"  K values: {k_values}",
        f"  Runtime:  {elapsed:.1f}s",
        "=" * 70,
        "",
        "--- AGGREGATE (macro-average over all queries) ---",
        "",
    ]

    # Print aggregate metrics grouped by type
    for k in k_values:
        lines.append(f"  Recall@{k:<4}  {aggregate.get(f'recall@{k}', 0):.4f}")
    lines.append("")
    for k in k_values:
        lines.append(f"  Prec@{k:<4}    {aggregate.get(f'precision@{k}', 0):.4f}")
    lines.append("")
    lines.append(f"  MRR          {aggregate.get('mrr', 0):.4f}")
    lines.append("")
    for k in k_values:
        lines.append(f"  NDCG@{k:<4}    {aggregate.get(f'ndcg@{k}', 0):.4f}")

    lines.append("")
    lines.append("--- PER-QUERY RESULTS ---")

    for i, q in enumerate(per_query, start=1):
        lines.append("")
        lines.append(f"  Query {i}: {q['query'][:80]}")
        lines.append(f"    Retrieved: {q['retrieved_count']} nodes")
        lines.append(f"    Relevant:  {q['relevant_count']} ground truth")
        lines.append(f"    Matched:   {q['matched_count']} hits")
        for k in k_values:
            lines.append(
                f"    R@{k}={q['metrics'].get(f'recall@{k}', 0):.3f}  "
                f"P@{k}={q['metrics'].get(f'precision@{k}', 0):.3f}  "
                f"NDCG@{k}={q['metrics'].get(f'ndcg@{k}', 0):.3f}"
            )
        lines.append(f"    MRR={q['metrics'].get('mrr', 0):.3f}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Main evaluation loop
# ══════════════════════════════════════════════════════════════

def run_evaluation(dataset_path: str, pdf_override: Optional[List[str]] = None,
                   k_override: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Run the full RAG evaluation pipeline.

    Args:
        dataset_path:  Path to the JSON test dataset (queries + keywords).
        pdf_override:  If provided, overrides pdf_paths from the JSON.
        k_override:    If provided, overrides k_values from the JSON.

    Returns:
        Dict with 'per_query', 'aggregate', 'metadata' keys.
    """
    print(f"\n{'=' * 60}")
    print("  RAG Evaluation Suite")
    print(f"{'=' * 60}\n")

    # 1. Load dataset
    print(f"[1/4] Loading test dataset: {dataset_path}")
    dataset = _load_dataset(dataset_path)
    k_values = k_override or dataset.get("k_values", [3, 5, 10])
    queries = dataset["queries"]
    pdf_paths = pdf_override or dataset.get("pdf_paths", [])
    print(f"       {len(queries)} queries, K={k_values}")

    # 2. Initialize pipeline
    print("[2/4] Initializing RAG pipeline...")
    t_total = time.perf_counter()
    pipeline = RAGPipeline()

    # Set extraction output to test output dir
    extraction_dir = str(OUTPUT_DIR / "extraction")
    pipeline.set_extraction_dir(extraction_dir)

    # 3. Ingest PDFs if provided
    if pdf_paths:
        print(f"[3/4] Ingesting {len(pdf_paths)} PDFs...")
        ingest_result = pipeline.ingest(pdf_paths)
        if not ingest_result.success:
            print(f"  WARNING: Ingest failed -- {ingest_result.error}")
            print("  Continuing with existing index data (if any)...")
        else:
            d = ingest_result.data
            assert d is not None
            print(f"       Ingested {d['chunks']} chunks, {d['images']} images "
                  f"from {d['success']}/{d['total']} files")
    else:
        print("[3/4] No PDFs to ingest (using existing index data)")

    # 4. Run queries and compute metrics
    print(f"[4/4] Running {len(queries)} evaluation queries...\n")
    per_query_results: List[Dict[str, Any]] = []
    all_metric_dicts: List[Dict[str, float]] = []

    for i, q in enumerate(queries, start=1):
        query_text = q["query"]
        ground_truth_ids = q.get("relevant_ids", [])
        keywords = q.get("relevant_keywords", [])
        graded_relevance = q.get("graded_relevance", None)

        print(f"  [{i}/{len(queries)}] \"{query_text[:60]}\" ... ", end="", flush=True)

        # Run query
        t_q = time.perf_counter()
        result: RAGResult = pipeline.query(query_text)
        query_time = time.perf_counter() - t_q

        if not result.success:
            print(f"FAILED ({result.error})")
            per_query_results.append({
                "query": query_text,
                "success": False,
                "error": result.error,
                "retrieved_count": 0,
                "relevant_count": len(ground_truth_ids),
                "matched_count": 0,
                "metrics": {f"recall@{k}": 0.0 for k in k_values}
                         | {f"precision@{k}": 0.0 for k in k_values}
                         | {f"ndcg@{k}": 0.0 for k in k_values}
                         | {"mrr": 0.0},
                "query_time_sec": round(query_time, 3),
            })
            all_metric_dicts.append(per_query_results[-1]["metrics"])
            continue

        assert result.data is not None
        context: RAGContext = result.data

        # Extract retrieved IDs from pipeline output
        retrieved_ids, id_to_content = _extract_retrieved_ids(context, keywords)

        # Resolve which IDs are relevant
        relevant_ids = _resolve_relevant_ids(
            retrieved_ids, id_to_content, ground_truth_ids, keywords
        )

        # Compute metrics
        metrics = compute_all_metrics(
            retrieved_ids, relevant_ids, k_values, graded_relevance
        )
        all_metric_dicts.append(metrics)

        # Count matched hits in top-max(k)
        max_k = max(k_values) if k_values else 10
        matched = len(set(retrieved_ids[:max_k]) & set(relevant_ids))

        per_query_results.append({
            "query": query_text,
            "success": True,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant_ids),
            "matched_count": matched,
            "retrieved_ids": retrieved_ids[:max_k],
            "relevant_ids": relevant_ids,
            "metrics": metrics,
            "query_time_sec": round(query_time, 3),
            "assembled_prompt_preview": context.assembled_prompt[:200],
        })

        # Print inline result
        r_at_k = metrics.get(f"recall@{k_values[-1]}", 0)
        p_at_k = metrics.get(f"precision@{k_values[0]}", 0)
        mrr = metrics.get("mrr", 0)
        print(f"R@{k_values[-1]}={r_at_k:.2f}  P@{k_values[0]}={p_at_k:.2f}  MRR={mrr:.2f}  ({query_time:.2f}s)")

    # 5. Aggregate
    aggregate = average_metrics(all_metric_dicts)
    elapsed = time.perf_counter() - t_total

    # 6. Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON output (full detail)
    json_path = OUTPUT_DIR / f"eval_results_{timestamp}.json"
    output_data = {
        "metadata": {
            "input_path": str(dataset_path),
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(queries),
            "k_values": k_values,
            "pdf_paths": pdf_paths,
            "total_runtime_sec": round(elapsed, 2),
        },
        "aggregate": {k: round(v, 4) for k, v in aggregate.items()},
        "per_query": per_query_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  JSON results  -> {json_path}")

    # Text summary
    summary_text = _format_summary_table(per_query_results, aggregate, k_values, elapsed)
    txt_path = OUTPUT_DIR / f"eval_summary_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"  Text summary  -> {txt_path}")

    # Also print to console
    print()
    print(summary_text)

    return output_data


# ══════════════════════════════════════════════════════════════
#  Public entry point (called by test_metrics.py)
# ══════════════════════════════════════════════════════════════

def run_evaluation_with_pdfs(
    pdf_paths: List[str],
    dataset_path: str,
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper called by test_metrics.py --pdf.

    Args:
        pdf_paths:    List of absolute PDF file paths to ingest.
        dataset_path: Path to queries JSON (sample_dataset.json by default).
        k_values:     K cut-offs. Default: [3, 5, 10].
    """
    return run_evaluation(
        dataset_path=dataset_path,
        pdf_override=pdf_paths,
        k_override=k_values,
    )


def run_evaluation_on_pipeline(
    pipeline,
    dataset_path: str,
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run evaluation queries on an already-initialized and ingested pipeline.
    Called by test_metrics.py --interactive after ingestion is done.
    """
    print(f"\n{'=' * 60}")
    print("  RAG Evaluation (on existing pipeline)")
    print(f"{'=' * 60}\n")

    # 1. Load dataset
    print(f"[1/2] Loading test dataset: {dataset_path}")
    dataset = _load_dataset(dataset_path)
    k_vals = k_values or dataset.get("k_values", [3, 5, 10])
    queries = dataset["queries"]
    print(f"       {len(queries)} queries, K={k_vals}")

    # 2. Run queries and compute metrics
    print(f"[2/2] Running {len(queries)} evaluation queries...\n")
    t_total = time.perf_counter()
    per_query_results: List[Dict[str, Any]] = []
    all_metric_dicts: List[Dict[str, float]] = []

    for i, q in enumerate(queries, start=1):
        query_text = q["query"]
        ground_truth_ids = q.get("relevant_ids", [])
        keywords = q.get("relevant_keywords", [])
        graded_relevance = q.get("graded_relevance", None)

        print(f"  [{i}/{len(queries)}] \"{query_text[:60]}\" ... ", end="", flush=True)

        t_q = time.perf_counter()
        result: RAGResult = pipeline.query(query_text)
        query_time = time.perf_counter() - t_q

        if not result.success:
            print(f"FAILED ({result.error})")
            per_query_results.append({
                "query": query_text, "success": False, "error": result.error,
                "retrieved_count": 0, "relevant_count": len(ground_truth_ids),
                "matched_count": 0,
                "metrics": {f"recall@{k}": 0.0 for k in k_vals}
                         | {f"precision@{k}": 0.0 for k in k_vals}
                         | {f"ndcg@{k}": 0.0 for k in k_vals}
                         | {"mrr": 0.0},
                "query_time_sec": round(query_time, 3),
            })
            all_metric_dicts.append(per_query_results[-1]["metrics"])
            continue

        assert result.data is not None
        context: RAGContext = result.data
        retrieved_ids, id_to_content = _extract_retrieved_ids(context, keywords)
        relevant_ids = _resolve_relevant_ids(
            retrieved_ids, id_to_content, ground_truth_ids, keywords
        )
        metrics = compute_all_metrics(retrieved_ids, relevant_ids, k_vals, graded_relevance)
        all_metric_dicts.append(metrics)

        max_k = max(k_vals) if k_vals else 10
        matched = len(set(retrieved_ids[:max_k]) & set(relevant_ids))

        per_query_results.append({
            "query": query_text, "success": True,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant_ids),
            "matched_count": matched,
            "retrieved_ids": retrieved_ids[:max_k],
            "relevant_ids": relevant_ids,
            "metrics": metrics,
            "query_time_sec": round(query_time, 3),
            "assembled_prompt_preview": context.assembled_prompt[:200],
        })

        r_at_k = metrics.get(f"recall@{k_vals[-1]}", 0)
        p_at_k = metrics.get(f"precision@{k_vals[0]}", 0)
        mrr = metrics.get("mrr", 0)
        print(f"R@{k_vals[-1]}={r_at_k:.2f}  P@{k_vals[0]}={p_at_k:.2f}  MRR={mrr:.2f}  ({query_time:.2f}s)")

    # Aggregate & write
    aggregate = average_metrics(all_metric_dicts)
    elapsed = time.perf_counter() - t_total

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUT_DIR / f"eval_results_{timestamp}.json"
    output_data = {
        "metadata": {
            "input_path": str(dataset_path),
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(queries),
            "k_values": k_vals,
            "total_runtime_sec": round(elapsed, 2),
        },
        "aggregate": {k: round(v, 4) for k, v in aggregate.items()},
        "per_query": per_query_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  JSON results  -> {json_path}")

    summary_text = _format_summary_table(per_query_results, aggregate, k_vals, elapsed)
    txt_path = OUTPUT_DIR / f"eval_summary_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"  Text summary  -> {txt_path}")
    print()
    print(summary_text)

    return output_data
