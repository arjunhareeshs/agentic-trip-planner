"""
parallel_retriever.py — Parallel multi-index retrieval across text, image, and captions.

Implements RetrieverProtocol.
Orchestrates parallel calls to the 3 indices using ThreadPoolExecutor.
Handles partial failures gracefully (one index failing doesn't kill the request).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np

from ..protocols import RetrieverProtocol
from ..rag_types import RetrievalCandidate
from ..utils.exceptions import RetrievalError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ParallelRetriever(RetrieverProtocol):
    """
    Executes synchronous parallel retrieval across multiple vector indices
    using ThreadPoolExecutor (no asyncio needed).

    Args:
        text_index: IndexProtocol instance for text.
        image_index: IndexProtocol instance for images.
        caption_index: IndexProtocol instance for captions.
    """

    def __init__(self, text_index, image_index, caption_index):
        self._text_index = text_index
        self._image_index = image_index
        self._caption_index = caption_index

    def retrieve(
        self,
        text_embedding: np.ndarray,
        clip_embedding: np.ndarray,
        filters: Optional[Any] = None,
        top_k_map: Optional[Dict[str, int]] = None,
    ) -> List[RetrievalCandidate]:
        """
        Run parallel retrieval (synchronous, thread-parallelized).

        Args:
            text_embedding: BGE embedding for text/caption search.
            clip_embedding: CLIP embedding for image search.
            filters: LlamaIndex MetadataFilters.
            top_k_map: dict mapping index name to top_k.

        Returns:
            Merged list of RetrievalCandidates.
        """
        if top_k_map is None:
            top_k_map = {"text": 15, "image": 10, "caption": 10}

        t0 = time.perf_counter()
        results: Dict[str, List[RetrievalCandidate]] = {}
        errors: Dict[str, str] = {}

        # Use ThreadPoolExecutor for parallel I/O-bound vector search calls
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="retriever") as pool:
            futures = {
                pool.submit(
                    self._safe_query, self._text_index, text_embedding, filters, top_k_map.get("text", 15)
                ): "text",
                pool.submit(
                    self._safe_query, self._image_index, clip_embedding, filters, top_k_map.get("image", 10)
                ): "image",
                pool.submit(
                    self._safe_query, self._caption_index, text_embedding, filters, top_k_map.get("caption", 10)
                ): "caption",
            }

            for future in as_completed(futures):
                source = futures[future]
                try:
                    results[source] = future.result()
                except Exception as exc:
                    errors[source] = str(exc)
                    logger.error("Parallel search failed for %s: %s", source, exc)
                    results[source] = []

        if len(errors) == 3:
            raise RetrievalError("All 3 indices failed during parallel retrieval.")

        # Merge results
        merged = []
        for source_results in results.values():
            merged.extend(source_results)

        elapsed = time.perf_counter() - t0
        logger.debug("Parallel retrieval completed in %.3fs (%d results)", elapsed, len(merged))
        return merged

    def _safe_query(self, index, embedding, filters, top_k) -> List[RetrievalCandidate]:
        """Query index if not empty, else return empty list."""
        if index.is_empty():
            return []
        return index.query(embedding, filters=filters, top_k=top_k)
