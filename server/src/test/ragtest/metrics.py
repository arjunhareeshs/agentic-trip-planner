"""
metrics.py — Information Retrieval metrics for RAG evaluation.

Implements:
  • Recall@K      — fraction of relevant docs found in top-K
  • Precision@K   — fraction of top-K that are relevant
  • MRR           — reciprocal rank of the first relevant result
  • NDCG@K        — normalized discounted cumulative gain

All functions operate on lists of retrieved node_ids vs. ground-truth
relevant node_ids (with optional graded relevance for NDCG).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:k]| / |relevant|

    Args:
        retrieved_ids: Ordered list of retrieved document/node IDs.
        relevant_ids:  Ground-truth relevant IDs (unordered).
        k: Cut-off rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k & relevant_set) / len(relevant_set)


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:k]| / K

    Args:
        retrieved_ids: Ordered list of retrieved document/node IDs.
        relevant_ids:  Ground-truth relevant IDs (unordered).
        k: Cut-off rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if k <= 0.
    """
    if k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k & relevant_set) / k


def mean_reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> float:
    """
    MRR = 1 / rank_of_first_relevant_result

    Args:
        retrieved_ids: Ordered list of retrieved document/node IDs.
        relevant_ids:  Ground-truth relevant IDs (unordered).

    Returns:
        Float in [0, 1]. Returns 0.0 if no relevant result found.
    """
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
    graded_relevance: Optional[Dict[str, float]] = None,
) -> float:
    """
    Normalized Discounted Cumulative Gain @ K.

    If graded_relevance is provided, uses those scores (higher = more relevant).
    Otherwise uses binary relevance (1 if in relevant_ids, else 0).

    Args:
        retrieved_ids:    Ordered list of retrieved document/node IDs.
        relevant_ids:     Ground-truth relevant IDs.
        k:                Cut-off rank.
        graded_relevance: Optional dict {node_id: relevance_score}.
                          Default binary: relevant=1, non-relevant=0.

    Returns:
        Float in [0, 1]. Returns 0.0 if no relevant docs exist.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    # Build relevance lookup
    if graded_relevance:
        rel_lookup = graded_relevance
    else:
        rel_lookup = {doc_id: 1.0 for doc_id in relevant_ids}

    # DCG for the retrieved ranking
    dcg = _dcg(retrieved_ids[:k], rel_lookup)

    # Ideal DCG: sort all relevant docs by relevance descending
    ideal_order = sorted(rel_lookup.keys(), key=lambda d: rel_lookup[d], reverse=True)[:k]
    idcg = _dcg(ideal_order, rel_lookup)

    return dcg / idcg if idcg > 0 else 0.0


def _dcg(ranked_ids: List[str], rel_lookup: Dict[str, float]) -> float:
    """Compute Discounted Cumulative Gain for a ranked list."""
    score = 0.0
    for i, doc_id in enumerate(ranked_ids):
        rel = rel_lookup.get(doc_id, 0.0)
        # Standard formula: (2^rel - 1) / log2(rank + 1)
        score += (math.pow(2, rel) - 1) / math.log2(i + 2)  # i+2 because rank starts at 1
    return score


# ── Aggregation helpers ──────────────────────────────────────

def compute_all_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int],
    graded_relevance: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute all 4 metrics for multiple K values in one call.

    Returns:
        Dict like:
        {
            "recall@3": 0.66, "recall@5": 1.0,
            "precision@3": 0.33, "precision@5": 0.2,
            "mrr": 0.5,
            "ndcg@3": 0.72, "ndcg@5": 0.85,
        }
    """
    results: Dict[str, float] = {}

    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k, graded_relevance)

    results["mrr"] = mean_reciprocal_rank(retrieved_ids, relevant_ids)

    return results


def average_metrics(all_results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute macro-average across multiple queries.

    Args:
        all_results: List of per-query metric dicts from compute_all_metrics().

    Returns:
        Dict with same keys, values averaged.
    """
    if not all_results:
        return {}

    keys = all_results[0].keys()
    averaged = {}
    for key in keys:
        values = [r[key] for r in all_results if key in r]
        averaged[key] = sum(values) / len(values) if values else 0.0

    return averaged
