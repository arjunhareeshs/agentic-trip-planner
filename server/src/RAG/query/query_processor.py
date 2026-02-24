"""
query_processor.py — Main entry point for the human-readable query processing pipeline.

Implements QueryProcessorProtocol.
Orchestrates:
  1. Embedding the query (BGE + CLIP)
  2. Building pre-filters (MetadataFilterBuilder)
  3. Parallel retrieval (ParallelRetriever)
  4. Deduplication and sorting
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from ..protocols import QueryProcessorProtocol
from ..rag_types import ProcessedQuery, RetrievalCandidate
from ..utils.exceptions import QueryError, RetrievalError
from ..utils.logger import get_logger
from .metadata_filter import MetadataFilterBuilder
from .parallel_retriever import ParallelRetriever

logger = get_logger(__name__)


class QueryProcessor(QueryProcessorProtocol):
    """
    Orchestrator for the query processing stage.
    Uses synchronous thread-based parallel retrieval (no asyncio needed).
    """

    def __init__(
        self,
        text_index,
        image_index,
        caption_index,
        text_embedder,
        image_embedder,
        metadata_extractor,
        retrieval_config=None,
    ):
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self._config = retrieval_config

        # Sub-components
        self._filter_builder = MetadataFilterBuilder()
        self._retriever = ParallelRetriever(
            text_index, image_index, caption_index
        )

    # ── QueryProcessorProtocol ────────────────────────────────

    def process(self, query: ProcessedQuery) -> List[RetrievalCandidate]:
        """
        Synchronous query processing — no asyncio overhead.
        """
        t0 = time.perf_counter()
        logger.info("QueryProcessor: processing query '%s'", query.text[:80])

        top_k_map = {
            "text": getattr(self._config, "text_top_k", 15),
            "image": getattr(self._config, "image_top_k", 10),
            "caption": getattr(self._config, "caption_top_k", 10),
        }

        # 1. Build metadata pre-filter
        meta_filter = self._filter_builder.build_filters(query.detected_metadata)

        # 2. Compute query embeddings
        text_embedding = self._embed_for_text(query.text)
        clip_embedding = self._embed_for_image(query.text)

        # 3. Retrieve (synchronous thread-based parallel)
        try:
            candidates = self._retriever.retrieve(
                text_embedding=text_embedding,
                clip_embedding=clip_embedding,
                filters=meta_filter,
                top_k_map=top_k_map,
            )
        except Exception as exc:
            logger.error("Retrieval stage failed: %s", exc)
            raise RetrievalError(f"Query retrieval failed: {exc}")

        # 4. Deduplicate, sort, and return
        unique_candidates = self._merge_and_deduplicate(candidates)

        elapsed = time.perf_counter() - t0
        logger.info(
            "QueryProcessor: retrieved %d unique candidates in %.2fs",
            len(unique_candidates), elapsed,
        )
        return unique_candidates

    # ── Embedding helpers ─────────────────────────────────────

    def _embed_for_text(self, query_text: str) -> np.ndarray:
        """Embed query in BGE space for text + caption indices."""
        try:
            return self._text_embedder.embed_single(query_text)
        except Exception as exc:
            logger.error("Text query embedding failed: %s -- using zero vector.", exc)
            return np.zeros(768, dtype=np.float32)

    def _embed_for_image(self, query_text: str) -> np.ndarray:
        """Embed query in CLIP text space for image index."""
        try:
            return self._image_embedder.embed_text(query_text)
        except Exception as exc:
            logger.error("CLIP query embedding failed: %s -- using zero vector.", exc)
            return np.zeros(512, dtype=np.float32)

    # ── Deduplication ─────────────────────────────────────────

    def _merge_and_deduplicate(
        self,
        candidates: List[RetrievalCandidate],
    ) -> List[RetrievalCandidate]:
        """
        Merge candidates, keeping highest score per node_id.
        Sort descending by similarity_score.
        """
        best: Dict[str, RetrievalCandidate] = {}
        for c in candidates:
            if c.node_id not in best or c.similarity_score > best[c.node_id].similarity_score:
                best[c.node_id] = c

        return sorted(best.values(), key=lambda c: c.similarity_score, reverse=True)
