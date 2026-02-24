"""
router.py — Query routing for complex RAG tasks.

Responsible for deciding which retrieval engine to use based on query intent.
Currently a passthrough for the standard parallel retriever, but architected
to support SubQuestionQueryEngine or Tool-use routing in the future.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..protocols import RetrieverProtocol
from ..rag_types import ProcessedQuery, RetrievalCandidate


class QueryRouter:
    """
    Routes queries to the appropriate retrieval mechanism.
    """

    def __init__(self, primary_retriever: RetrieverProtocol):
        self._retriever = primary_retriever

    def route_and_retrieve(
        self,
        query: ProcessedQuery,
        text_embedding: Any,
        clip_embedding: Any,
        filters: Optional[Any] = None,
        top_k_map: Optional[dict] = None,
    ) -> List[RetrievalCandidate]:
        """
        Route the query and execute retrieval.

        This currently always uses the primary_retriever, but allows for
        adding intent-based routing logic (e.g. 'search images only' or
        'summarize everything') without changing the QueryProcessor.
        """
        return self._retriever.retrieve(
            text_embedding=text_embedding,
            clip_embedding=clip_embedding,
            filters=filters,
            top_k_map=top_k_map,
        )
