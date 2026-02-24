"""
text_embedder.py — BAAI BGE-base-en-v1.5 text embedding wrapper.

Implements TextEmbedderProtocol.
  • Batch embedding with configurable batch size
  • Auto-truncation to max sequence length (512 tokens)
  • L2-normalized output vectors (768-dim)
  • LlamaIndex-compatible BaseEmbedding wrapper

ISOLATION: Imports ONLY from protocols, utils, embedding.model_manager.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

from ..protocols import TextEmbedderProtocol
from ..utils.exceptions import EmbeddingError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BGETextEmbedder(TextEmbedderProtocol):
    """BAAI BGE-base-en-v1.5 text embedder."""

    def __init__(self, model_manager, config=None):
        self._mm = model_manager
        self._dim = getattr(config, "dimension", 768)
        self._batch_size = getattr(config, "batch_size", 32)
        self._max_seq_length = getattr(config, "max_seq_length", 512)
        self._normalize = getattr(config, "normalize", True)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, 768) L2-normalized float32 array."""
        if not texts:
            raise EmbeddingError("embed_batch received an empty list.")

        cleaned = self._preprocess(texts)
        t0 = time.perf_counter()
        try:
            model = self._mm.get_text_model()
            embeddings = model.encode(
                cleaned,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"BGE embedding failed: {exc}") from exc

        if embeddings.shape[-1] != self._dim:
            raise EmbeddingError(
                f"BGE dimension mismatch: expected {self._dim}, got {embeddings.shape[-1]}"
            )

        logger.debug("Embedded %d texts in %.2fs (BGE)", len(texts), time.perf_counter() - t0)
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns shape (768,)."""
        return self.embed_batch([text])[0]

    def _preprocess(self, texts: List[str]) -> List[str]:
        """Validate, clean, and truncate texts to max_seq_length words."""
        cleaned = []
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                text = "[empty]"
            words = text.split()
            if len(words) > self._max_seq_length:
                text = " ".join(words[: self._max_seq_length])
            cleaned.append(text.strip())
        return cleaned

    def as_llamaindex_embedding(self):
        """Return a LlamaIndex-compatible BaseEmbedding wrapper."""
        from llama_index.core.embeddings import BaseEmbedding

        outer = self

        class _LlamaIndexBGE(BaseEmbedding):
            def _get_query_embedding(self, query: str) -> List[float]:
                return outer.embed_single(query).tolist()

            def _get_text_embedding(self, text: str) -> List[float]:
                return outer.embed_single(text).tolist()

            async def _aget_query_embedding(self, query: str) -> List[float]:
                return self._get_query_embedding(query)

            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                return outer.embed_batch(texts).tolist()

        return _LlamaIndexBGE()
