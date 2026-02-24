"""
image_index.py — LlamaIndex VectorStoreIndex for image CLIP vectors.

Implements IndexProtocol for the image modality.
  • CLIP ViT-B/32 embeddings (512-dim)
  • Full ImageNodeData metadata per node (matches spec exactly)
  • Separate vector space from text index
  • Disk persistence with incremental add

ISOLATION: Imports ONLY from protocols, types, utils. No other RAG submodules.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..protocols import IndexProtocol
from ..rag_types import ImageNodeData, RetrievalCandidate
from ..utils.exceptions import IndexingError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImageVectorIndex(IndexProtocol):
    """
    Vector index for image CLIP embeddings.

    Args:
        image_embedder: CLIPImageEmbedder instance.
        config: indexing sub-section of RAGSettings.
        persist_dir: Absolute path for index persistence.
    """

    def __init__(self, image_embedder, config=None, persist_dir: str = "./indices"):
        self._embedder = image_embedder
        self._config = config
        self._persist_dir = Path(persist_dir) / "image_index"
        self._index = None
        self._is_empty_flag = True

    # ── IndexProtocol ─────────────────────────────────────────

    def build(self, nodes: List[ImageNodeData]) -> None:
        """Build image index from ImageNodeData list."""
        if not nodes:
            logger.warning("ImageVectorIndex.build() called with 0 nodes -- skipping.")
            return

        t0 = time.perf_counter()
        logger.info("Building image index with %d nodes...", len(nodes))

        try:
            from llama_index.core import VectorStoreIndex

            llama_nodes = [self._image_node_to_llama(n) for n in nodes]

            # We provide pre-computed embeddings — no re-embedding needed
            self._index = VectorStoreIndex(llama_nodes, show_progress=False)
            self._is_empty_flag = False

            elapsed = time.perf_counter() - t0
            logger.info("Image index built in %.2fs (%d nodes)", elapsed, len(nodes))

            if getattr(self._config, "auto_persist", True):
                self.persist()

        except Exception as exc:
            raise IndexingError(
                f"Image index build failed: {exc}",
                context={"n_nodes": len(nodes)},
            ) from exc

    def add(self, nodes: List[ImageNodeData]) -> None:
        """Incrementally add image nodes."""
        if self._index is None:
            return self.build(nodes)

        try:
            for node in nodes:
                llama_node = self._image_node_to_llama(node)
                self._index.insert(llama_node)

            if getattr(self._config, "auto_persist", True):
                self.persist()

        except Exception as exc:
            raise IndexingError(
                f"Image index add failed: {exc}",
                context={"n_nodes": len(nodes)},
            ) from exc

    def query(
        self,
        embedding: np.ndarray,
        filters: Optional[Any] = None,
        top_k: int = 10,
    ) -> List[RetrievalCandidate]:
        """CLIP-space similarity search."""
        if self._index is None or self._is_empty_flag:
            logger.warning("ImageVectorIndex.query() called on empty index.")
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
            logger.error("Image index query failed: %s", exc)
            return []

    def persist(self) -> None:
        if self._index is None:
            return
        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._index.storage_context.persist(persist_dir=str(self._persist_dir))
            logger.debug("Image index persisted to %s", self._persist_dir)
        except Exception as exc:
            raise IndexingError(
                f"Image index persist failed: {exc}",
                context={"path": str(self._persist_dir)},
            ) from exc

    def load(self) -> None:
        try:
            from llama_index.core import StorageContext, load_index_from_storage

            storage_context = StorageContext.from_defaults(
                persist_dir=str(self._persist_dir)
            )
            self._index = load_index_from_storage(storage_context)
            self._is_empty_flag = False
            logger.info("Image index loaded from %s", self._persist_dir)
        except Exception as exc:
            raise IndexingError(
                f"Image index load failed: {exc}",
                context={"path": str(self._persist_dir)},
            ) from exc

    def is_empty(self) -> bool:
        return self._is_empty_flag

    # ── Helpers ───────────────────────────────────────────────

    def _image_node_to_llama(self, node: ImageNodeData):
        """
        Convert ImageNodeData to a LlamaIndex TextNode with pre-computed embedding.
        The node text is the caption; the embedding is the CLIP image vector.
        """
        from llama_index.core.schema import TextNode

        return TextNode(
            id_=node.image_id,
            text=node.caption or f"Image from {node.source_pdf} page {node.page_number}",
            embedding=list(node.image_vector),   # pre-computed CLIP embedding
            metadata={
                "node_type": "image",
                "caption": node.caption,
                "city": node.city,
                "scene_type": node.scene_type,
                "crowd_level": node.crowd_level,
                "lighting": node.lighting,
                "emotion_tags": list(node.emotion_tags),
                "source_pdf": node.source_pdf,
                "page_number": node.page_number,
                "image_path": node.image_path,
                "image_id": node.image_id,
            },
        )

    def _node_to_candidate(self, scored_node) -> RetrievalCandidate:
        node = scored_node.node
        return RetrievalCandidate(
            node_id=node.node_id,
            content=node.get_content(),
            similarity_score=float(scored_node.score or 0.0),
            source_index="image",
            node_type="image",
            metadata=node.metadata or {},
        )
