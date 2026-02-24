"""
text_index.py — LlamaIndex VectorStoreIndex for text chunks.

Implements IndexProtocol for the text modality.
  • BGE-base-en-v1.5 embeddings (768-dim)
  • Node metadata: source_pdf, page_number, section_hierarchy, element_type, city, scene_type
  • Disk persistence with incremental add support
  • Fully swappable backend (default: in-memory SimpleVectorStore → pinecone/weaviate with config change)

ISOLATION: Imports ONLY from protocols, types, utils, and LlamaIndex. No other RAG submodules.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..protocols import IndexProtocol
from ..rag_types import RetrievalCandidate, StructuredChunk
from ..utils.exceptions import IndexingError
from ..utils.logger import get_logger
from ..utils.validators import validate_chunks

logger = get_logger(__name__)


class TextVectorIndex(IndexProtocol):
    """
    Vector index for document text chunks.

    Args:
        text_embedder: BGETextEmbedder instance.
        config: indexing sub-section of RAGSettings.
        persist_dir: Absolute path for index persistence.
    """

    def __init__(self, text_embedder, config=None, persist_dir: str = "./indices"):
        self._embedder = text_embedder
        self._config = config
        self._persist_dir = Path(persist_dir) / "text_index"
        self._index = None   # Lazy: LlamaIndex VectorStoreIndex
        self._is_empty_flag = True

    # ── IndexProtocol ─────────────────────────────────────────

    @validate_chunks(arg_name="nodes")
    def build(self, nodes: List[StructuredChunk]) -> None:
        """Build the index from a list of StructuredChunk objects."""
        t0 = time.perf_counter()
        logger.info("Building text index with %d chunks...", len(nodes))

        try:
            from llama_index.core import VectorStoreIndex
            from llama_index.core.schema import TextNode

            llama_nodes = [self._chunk_to_llama_node(chunk) for chunk in nodes]
            embed_model = self._embedder.as_llamaindex_embedding()

            self._index = VectorStoreIndex(
                llama_nodes,
                embed_model=embed_model,
                show_progress=False,
            )
            self._is_empty_flag = False

            elapsed = time.perf_counter() - t0
            logger.info("Text index built in %.2fs (%d nodes)", elapsed, len(nodes))

            if getattr(self._config, "auto_persist", True):
                self.persist()

        except Exception as exc:
            raise IndexingError(
                f"Text index build failed: {exc}",
                context={"n_chunks": len(nodes), "error": str(exc)},
            ) from exc

    def add(self, nodes: List[StructuredChunk]) -> None:
        """Incrementally add chunks to an existing index."""
        if self._index is None:
            return self.build(nodes)

        try:
            from llama_index.core.schema import TextNode

            llama_nodes = [self._chunk_to_llama_node(chunk) for chunk in nodes]
            for node in llama_nodes:
                self._index.insert(node)

            logger.info("Added %d chunks to text index.", len(nodes))
            if getattr(self._config, "auto_persist", True):
                self.persist()

        except Exception as exc:
            raise IndexingError(
                f"Text index incremental add failed: {exc}",
                context={"n_nodes": len(nodes)},
            ) from exc

    def query(
        self,
        embedding: np.ndarray,
        filters: Optional[Any] = None,
        top_k: int = 15,
    ) -> List[RetrievalCandidate]:
        """
        Vector similarity search on the text index.

        Returns:
            List of RetrievalCandidate sorted by similarity descending.
            Empty list if index is empty or no results found.
        """
        if self._index is None or self._is_empty_flag:
            logger.warning("TextVectorIndex.query() called on empty index.")
            return []

        try:
            retriever = self._index.as_retriever(
                similarity_top_k=top_k,
                filters=filters,
            )
            # Convert embedding to query — LlamaIndex handles vector-based retrieval
            query_bundle = self._build_query_bundle(embedding)
            nodes = retriever.retrieve(query_bundle)

            return [self._node_to_candidate(n) for n in nodes]

        except Exception as exc:
            logger.error("Text index query failed: %s", exc)
            return []   # Return empty — don't crash parallel retrieval

    def persist(self) -> None:
        """Save index to disk."""
        if self._index is None:
            return
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._index.storage_context.persist(persist_dir=str(self._persist_dir))
            logger.debug("Text index persisted to %s", self._persist_dir)
        except Exception as exc:
            raise IndexingError(
                f"Text index persist failed: {exc}",
                context={"path": str(self._persist_dir)},
            ) from exc

    def load(self) -> None:
        """Load index from disk."""
        try:
            from llama_index.core import StorageContext, load_index_from_storage

            storage_context = StorageContext.from_defaults(
                persist_dir=str(self._persist_dir)
            )
            embed_model = self._embedder.as_llamaindex_embedding()
            self._index = load_index_from_storage(
                storage_context, embed_model=embed_model
            )
            self._is_empty_flag = False
            logger.info("Text index loaded from %s", self._persist_dir)
        except Exception as exc:
            raise IndexingError(
                f"Text index load failed: {exc}",
                context={"path": str(self._persist_dir)},
            ) from exc

    def is_empty(self) -> bool:
        return self._is_empty_flag

    # ── Helpers ───────────────────────────────────────────────

    def _chunk_to_llama_node(self, chunk: StructuredChunk):
        """Convert a StructuredChunk to a LlamaIndex TextNode."""
        from llama_index.core.schema import TextNode

        meta = chunk.metadata
        return TextNode(
            id_=chunk.chunk_id,
            text=chunk.content,
            metadata={
                "source_pdf": meta.source_pdf,
                "page_number": meta.page_number,
                "section_hierarchy": list(meta.section_hierarchy),
                "element_type": meta.element_type,
                "city": meta.city,
                "scene_type": meta.scene_type,
                "crowd_level": meta.crowd_level,
                "lighting": meta.lighting,
                "emotion_tags": list(meta.emotion_tags),
                "chunk_index": chunk.chunk_index,
                "node_type": "text",
            },
        )

    def _build_query_bundle(self, embedding: np.ndarray):
        """Wrap a raw embedding into a LlamaIndex QueryBundle."""
        from llama_index.core.schema import QueryBundle

        return QueryBundle(
            query_str="",
            embedding=embedding.tolist(),
        )

    def _node_to_candidate(self, scored_node) -> RetrievalCandidate:
        """Convert a LlamaIndex NodeWithScore to RetrievalCandidate."""
        node = scored_node.node
        return RetrievalCandidate(
            node_id=node.node_id,
            content=node.get_content(),
            similarity_score=float(scored_node.score or 0.0),
            source_index="text",
            node_type="text",
            metadata=node.metadata or {},
        )
