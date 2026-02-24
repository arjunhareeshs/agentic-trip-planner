"""
ranking/cross_encoder_reranker.py — Cross-encoder reranker over combined text + image candidates.

Implements RankerProtocol.
  • Uses cross-encoder/ms-marco-MiniLM-L-6-v2 for relevance scoring
  • Query paired with: chunk content (for text nodes) or caption (for image nodes)
  • Reranked candidates sorted descending by cross-encoder score
  • Graceful fallback: if cross-encoder fails → returns candidates sorted by similarity
"""

from __future__ import annotations

import time
from typing import List

from ..protocols import RankerProtocol
from ..rag_types import RetrievalCandidate
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(RankerProtocol):
    """
    Cross-encoder reranker for combined retrieval candidates.

    Args:
        model_manager: ModelManager providing the cross-encoder.
        config: (optional) RankingConfig — currently not defined; reserved for future.
    """

    def __init__(self, model_manager, config=None):
        self._mm = model_manager
        self._config = config

    # ── RankerProtocol ────────────────────────────────────────

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        """
        Rerank candidates using cross-encoder scores.

        Args:
            query: Raw query string (not embedded).
            candidates: Combined retrieval candidates from all indices.
            top_k: Number of candidates to return after reranking.

        Returns:
            Up to `top_k` RetrievalCandidates with updated rerank_score field.
        """
        if not candidates:
            logger.debug("Reranker: no candidates to rerank.")
            return []

        top_k = min(top_k, len(candidates))

        try:
            t0 = time.perf_counter()
            pairs = self._build_query_pairs(query, candidates)
            scored = self._run_cross_encoder(pairs, candidates)
            reranked = sorted(scored, key=lambda x: x.rerank_score, reverse=True)

            elapsed = time.perf_counter() - t0
            logger.info(
                "Reranked %d → top-%d candidates in %.2fs",
                len(candidates), top_k, elapsed,
            )
            return reranked[:top_k]

        except Exception as exc:
            # Graceful fallback — never let reranker crash the pipeline
            logger.warning(
                "Cross-encoder reranking failed (using fallback sort): %s", exc
            )
            return sorted(candidates, key=lambda c: c.similarity_score, reverse=True)[:top_k]

    # ── Helpers ───────────────────────────────────────────────

    def _build_query_pairs(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
    ) -> List[tuple]:
        """
        For each candidate, build a (query, passage) pair.
        Image nodes: use caption as the passage text.
        Text nodes: use full content as the passage text.
        """
        pairs = []
        for c in candidates:
            if c.node_type == "image":
                passage = c.metadata.get("caption", c.content) or c.content
            else:
                passage = c.content

            # Truncate passage to avoid cross-encoder OOM
            passage = passage[:512] if len(passage) > 512 else passage
            pairs.append((query, passage))
        return pairs

    def _run_cross_encoder(
        self,
        pairs: List[tuple],
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """
        Run cross-encoder inference on all pairs.

        Returns candidates with updated rerank_score field.
        """
        cross_encoder = self._mm.get_cross_encoder()
        scores = cross_encoder.predict(pairs)   # returns numpy array

        updated = []
        for candidate, score in zip(candidates, scores):
            # Create a NEW RetrievalCandidate with the rerank_score
            updated_candidate = RetrievalCandidate(
                node_id=candidate.node_id,
                content=candidate.content,
                similarity_score=candidate.similarity_score,
                rerank_score=float(score),
                source_index=candidate.source_index,
                node_type=candidate.node_type,
                metadata=candidate.metadata,
            )
            updated.append(updated_candidate)

        return updated
