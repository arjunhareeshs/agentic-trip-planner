"""
caption_index.py — VectorStoreIndex for image caption text embeddings (BGE space).

Implements IndexProtocol for the caption modality.
  • BGE-base-en-v1.5 embeddings of caption text (768-dim)
  • Enables text-query → caption similarity (different from CLIP space)
  • Cross-references image nodes by image_id
  • Completes the 3-index parallel retrieval architecture

ISOLATION: Imports ONLY from protocols, types, utils. No other RAG submodules.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..protocols import IndexProtocol
from ..rag_types import ExtractedImage, ImageNodeData, RetrievalCandidate
from ..utils.exceptions import IndexingError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CaptionVectorIndex(IndexProtocol):
    """
    Vector index for caption text embeddings in BGE space.

    Args:
        text_embedder: BGETextEmbedder instance.
        config: indexing sub-section of RAGSettings.
        persist_dir: Absolute path for index persistence.
    """

    def __init__(self, text_embedder, config=None, persist_dir: str = "./indices"):
        self._embedder = text_embedder
        self._config = config
        self._persist_dir = Path(persist_dir) / "caption_index"
        self._index = None
        self._is_empty_flag = True

    # ── IndexProtocol ─────────────────────────────────────────

    def build(self, nodes: List[ImageNodeData]) -> None:
        """
        Build caption index from ImageNodeData.
        Uses the caption text embedded in BGE space.
        Skips nodes with empty captions.
        """
        captioned = [n for n in nodes if n.caption.strip()]
        if not captioned:
            logger.warning("CaptionVectorIndex.build(): all nodes have empty captions -- skipping.")
            return

        t0 = time.perf_counter()
        logger.info("Building caption index with %d captioned nodes...", len(captioned))

        try:
            from llama_index.core import VectorStoreIndex

            llama_nodes = self._build_caption_nodes(captioned)
            embed_model = self._embedder.as_llamaindex_embedding()

            self._index = VectorStoreIndex(
                llama_nodes,
                embed_model=embed_model,
                show_progress=False,
            )
            self._is_empty_flag = False

            elapsed = time.perf_counter() - t0
            logger.info("Caption index built in %.2fs (%d nodes)", elapsed, len(captioned))

            if getattr(self._config, "auto_persist", True):
                self.persist()

        except Exception as exc:
            raise IndexingError(
                f"Caption index build failed: {exc}",
                context={"n_captioned": len(captioned)},
            ) from exc

    def add(self, nodes: List[ImageNodeData]) -> None:
        captioned = [n for n in nodes if n.caption.strip()]
        if not captioned:
            return
        if self._index is None:
            return self.build(captioned)

        try:
            new_llama_nodes = self._build_caption_nodes(captioned)
            for node in new_llama_nodes:
                self._index.insert(node)

            if getattr(self._config, "auto_persist", True):
                self.persist()

        except Exception as exc:
            raise IndexingError(
                f"Caption index add failed: {exc}",
                context={"n_nodes": len(captioned)},
            ) from exc

    def query(
        self,
        embedding: np.ndarray,
        filters: Optional[Any] = None,
        top_k: int = 10,
    ) -> List[RetrievalCandidate]:
        """BGE-space text similarity search against caption embeddings."""
        if self._index is None or self._is_empty_flag:
            logger.warning("CaptionVectorIndex.query() called on empty index.")
            return []

        try:
            from llama_index.core.schema import QueryBundle

            retriever = self._index.as_retriever(
                similarity_top_k=top_k,
                filters=filters,
            )
            query_bundle = QueryBundle(
                query_str="",
                embedding=embedding.tolist(),
            )
            nodes = retriever.retrieve(query_bundle)
            return [self._node_to_candidate(n) for n in nodes]

        except Exception as exc:
            logger.error("Caption index query failed: %s", exc)
            return []

    def persist(self) -> None:
        if self._index is None:
            return
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._index.storage_context.persist(persist_dir=str(self._persist_dir))
            logger.debug("Caption index persisted to %s", self._persist_dir)
        except Exception as exc:
            raise IndexingError(
                f"Caption index persist failed: {exc}",
                context={"path": str(self._persist_dir)},
            ) from exc

    def load(self) -> None:
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
            logger.info("Caption index loaded from %s", self._persist_dir)
        except Exception as exc:
            raise IndexingError(
                f"Caption index load failed: {exc}",
                context={"path": str(self._persist_dir)},
            ) from exc

    def is_empty(self) -> bool:
        return self._is_empty_flag

    # ── Helpers ───────────────────────────────────────────────

    def _build_caption_nodes(self, image_nodes: List[ImageNodeData]):
        """Create TextNode per captioned image, using caption text for embedding."""
        from llama_index.core.schema import TextNode

        llama_nodes = []
        for node in image_nodes:
            llama_nodes.append(TextNode(
                id_=f"cap_{node.image_id}",
                text=node.caption,
                metadata={
                    "node_type": "caption",
                    "image_id": node.image_id,
                    "caption": node.caption,
                    "city": node.city,
                    "scene_type": node.scene_type,
                    "crowd_level": node.crowd_level,
                    "lighting": node.lighting,
                    "emotion_tags": list(node.emotion_tags),
                    "source_pdf": node.source_pdf,
                    "page_number": node.page_number,
                },
            ))
        return llama_nodes

    def _node_to_candidate(self, scored_node) -> RetrievalCandidate:
        node = scored_node.node
        return RetrievalCandidate(
            node_id=node.node_id,
            content=node.get_content(),
            similarity_score=float(scored_node.score or 0.0),
            source_index="caption",
            node_type="image",   # still an image node, just retrieved via caption
            metadata=node.metadata or {},
        )
