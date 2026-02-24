"""
test_metrics.py — RAG Evaluation Entry Point.

═══════════════════════════════════════════════════════════════
USAGE:
    cd server/src
    conda activate evenv

    # Unit tests only (no PDF needed):
    python -m test.ragtest.test_metrics

    # Full RAG evaluation with your PDF:
    python -m test.ragtest.test_metrics --pdf "C:/path/to/your/file.pdf"

    # Ingest PDF + open interactive query engine:
    python -m test.ragtest.test_metrics --pdf "file.pdf" --interactive

    # Multiple PDFs:
    python -m test.ragtest.test_metrics --pdf "doc1.pdf" --pdf "doc2.pdf"

    # Custom queries JSON (optional, defaults to sample_dataset.json):
    python -m test.ragtest.test_metrics --pdf "file.pdf" --dataset "my_queries.json"

    # Custom K values:
    python -m test.ragtest.test_metrics --pdf "file.pdf" -k 3 -k 5 -k 10

OUTPUT:
    Extraction output → vectordb/extraction_output/<pdf_name>/
      text/     — one file per page with all text elements
      tables/   — each table as a Markdown file
      images/   — extracted images

    Evaluation output → test/ragtest/output/
      eval_results_<timestamp>.json   — full per-query + aggregate
      eval_summary_<timestamp>.txt    — human-readable report
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from test.ragtest.metrics import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    compute_all_metrics,
    average_metrics,
)


# ══════════════════════════════════════════════════════════════
#  PHASE 1 — Metric Unit Tests
# ══════════════════════════════════════════════════════════════

def _approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) < tol


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = ["a", "c", "f"]
    assert _approx(recall_at_k(retrieved, relevant, 1), 1 / 3)
    assert _approx(recall_at_k(retrieved, relevant, 3), 2 / 3)
    assert _approx(recall_at_k(retrieved, relevant, 5), 2 / 3)
    assert _approx(recall_at_k(retrieved, [], 5), 0.0)
    print("  recall@K ........... PASS")


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = ["a", "c", "f"]
    assert _approx(precision_at_k(retrieved, relevant, 1), 1 / 1)
    assert _approx(precision_at_k(retrieved, relevant, 3), 2 / 3)
    assert _approx(precision_at_k(retrieved, relevant, 5), 2 / 5)
    assert _approx(precision_at_k(retrieved, relevant, 0), 0.0)
    print("  precision@K ........ PASS")


def test_mrr():
    assert _approx(mean_reciprocal_rank(["a", "b", "c"], ["a"]), 1.0)
    assert _approx(mean_reciprocal_rank(["x", "y", "a"], ["a", "b"]), 1 / 3)
    assert _approx(mean_reciprocal_rank(["x", "y", "z"], ["a"]), 0.0)
    print("  MRR ................ PASS")


def test_ndcg_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["a", "c"]
    assert 0.0 < ndcg_at_k(retrieved, relevant, 4) <= 1.0
    assert _approx(ndcg_at_k(["a", "c", "x", "y"], relevant, 2), 1.0)
    assert ndcg_at_k(["a", "c", "b", "d"], relevant, 4, {"a": 3.0, "c": 1.0}) > 0.0
    assert _approx(ndcg_at_k(retrieved, [], 4), 0.0)
    print("  NDCG@K ............. PASS")


def test_compute_all():
    metrics = compute_all_metrics(["a", "b", "c", "d", "e"], ["a", "c"], [3, 5])
    assert all(key in metrics for key in ["recall@3", "precision@5", "ndcg@3", "mrr"])
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    print("  compute_all ........ PASS")


def test_average():
    avg = average_metrics([{"recall@3": 0.5, "mrr": 1.0}, {"recall@3": 1.0, "mrr": 0.5}])
    assert _approx(avg["recall@3"], 0.75) and _approx(avg["mrr"], 0.75)
    print("  average_metrics .... PASS")


def run_unit_tests() -> bool:
    """Run all 6 metric unit tests. Returns True if all pass."""
    print("\n  PHASE 1 -- Metric Unit Tests")
    print("  " + "-" * 40)
    try:
        test_recall_at_k()
        test_precision_at_k()
        test_mrr()
        test_ndcg_at_k()
        test_compute_all()
        test_average()
        print("\n  ALL 6 METRIC TESTS PASSED\n")
        return True
    except AssertionError as exc:
        print(f"\n  TEST FAILED: {exc}\n")
        return False
    except Exception as exc:
        print(f"\n  TEST FAILED: {exc}\n")
        return False


# ══════════════════════════════════════════════════════════════
#  PHASE 2 — Full RAG Evaluation (only if --pdf is given)
# ══════════════════════════════════════════════════════════════

def run_rag_evaluation(pdf_paths: list, dataset_path: str, k_values: list):
    """
    Ingest the given PDFs, run evaluation queries, compute metrics,
    and write results to the output/ folder.
    """
    from test.ragtest.run_eval import run_evaluation_with_pdfs
    run_evaluation_with_pdfs(pdf_paths, dataset_path, k_values)


def run_interactive_mode(
    pdf_paths: list,
    dataset_path: str,
    k_values: list,
):
    """
    Ingest PDFs, run evaluation metrics, then start interactive query engine.
    Extraction output (text/images/tables) is written to test/ragtest/output/.
    """
    from RAG import RAGPipeline
    from RAG.output.query_engine import InteractiveQueryEngine

    # Determine output dir for extraction
    extraction_dir = str(_SCRIPT_DIR / "output" / "extraction")

    print("\n  Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    pipeline.set_extraction_dir(extraction_dir)

    print(f"  Ingesting {len(pdf_paths)} PDF(s)...")
    result = pipeline.ingest(pdf_paths)

    if not result.success:
        print(f"\n  INGEST FAILED: {result.error}")
        return

    d = result.data
    assert d is not None
    print(f"\n  Ingested {d['chunks']} chunks, {d['images']} images "
          f"from {d['success']}/{d['total']} files ({d['elapsed_sec']}s)")

    # Show extraction output locations
    for report in d.get("extraction", []):
        print(f"  Extraction output -> {report['output_dir']}")

    # Run evaluation metrics
    print(f"\n  PHASE 3 -- Running Evaluation")
    print("  " + "-" * 40)
    from test.ragtest.run_eval import run_evaluation_on_pipeline
    run_evaluation_on_pipeline(pipeline, dataset_path, k_values)

    # Start interactive query engine
    print(f"\n  PHASE 4 -- Interactive Query Engine")
    print("  " + "-" * 40)
    engine = InteractiveQueryEngine(pipeline)
    engine.run()


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAG Test Suite — unit tests + full pipeline evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unit tests only:
  python -m test.ragtest.test_metrics

  # Full evaluation with a PDF:
  python -m test.ragtest.test_metrics --pdf "C:/docs/travel_brochure.pdf"

  # Interactive query mode (ingest + ask questions):
  python -m test.ragtest.test_metrics --pdf doc.pdf --interactive

  # Multiple PDFs:
  python -m test.ragtest.test_metrics --pdf doc1.pdf --pdf doc2.pdf

  # Custom dataset + K values:
  python -m test.ragtest.test_metrics --pdf doc.pdf --dataset my_queries.json -k 3 -k 5
        """,
    )
    parser.add_argument(
        "--pdf", "-p",
        action="append",
        default=None,
        help="Path to a PDF file to ingest and evaluate. Can be repeated for multiple PDFs.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        default=False,
        help="After ingestion, open interactive query engine instead of running evaluation.",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=None,
        help="Path to queries JSON file. Defaults to sample_dataset.json.",
    )
    parser.add_argument(
        "-k",
        type=int,
        action="append",
        default=None,
        help="K value for @K metrics. Can be repeated. Default: 3, 5, 10.",
    )
    args = parser.parse_args()

    # Always run unit tests first
    passed = run_unit_tests()
    if not passed:
        sys.exit(1)

    # If --pdf given, run full RAG evaluation or interactive mode
    if args.pdf:
        # Validate all PDF paths exist
        for pdf in args.pdf:
            p = Path(pdf)
            if not p.exists():
                print(f"  ERROR: PDF not found: {pdf}")
                sys.exit(1)
            if p.suffix.lower() != ".pdf":
                print(f"  ERROR: Not a PDF file: {pdf}")
                sys.exit(1)

        dataset_path = args.dataset or str(_SCRIPT_DIR / "sample_dataset.json")
        k_values = args.k or [3, 5, 10]

        if args.interactive:
            # Interactive mode: ingest + eval + query REPL
            print("\n  PHASE 2 -- Ingest + Evaluate + Interactive")
            print("  " + "-" * 40)
            print(f"  PDFs:    {args.pdf}")
            print(f"  Dataset: {dataset_path}")
            print(f"  K:       {k_values}")
            run_interactive_mode(args.pdf, dataset_path, k_values)
        else:
            # Evaluation mode: ingest + metrics
            print("  PHASE 2 -- Full RAG Pipeline Evaluation")
            print("  " + "-" * 40)
            print(f"  PDFs:    {args.pdf}")
            print(f"  Dataset: {dataset_path}")
            print(f"  K:       {k_values}")
            print()

            run_rag_evaluation(args.pdf, dataset_path, k_values)
    else:
        print("  Tip: To run full RAG evaluation, add --pdf <path_to_your.pdf>")
        print("  Tip: For interactive mode, add --pdf <path> --interactive")
        print("  Example: python -m test.ragtest.test_metrics --pdf \"brochure.pdf\" --interactive")
        print()


if __name__ == "__main__":
    main()
