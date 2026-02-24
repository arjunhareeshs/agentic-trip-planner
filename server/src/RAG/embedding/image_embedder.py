"""
image_embedder.py — CLIP ViT-B/32 image and text embedding wrapper.

Implements ImageEmbedderProtocol.
  • embed_image(PIL.Image) → 512-dim CLIP vector
  • embed_text(str)        → 512-dim CLIP vector (for caption alignment)
  • L2-normalized output vectors

ISOLATION: Imports ONLY from protocols, utils, embedding.model_manager.
"""

from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
import torch

from ..protocols import ImageEmbedderProtocol
from ..utils.exceptions import EmbeddingError
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CLIPImageEmbedder(ImageEmbedderProtocol):
    """OpenAI CLIP ViT-B/32 embedder for both images and text."""

    def __init__(self, model_manager, config=None):
        self._mm = model_manager
        self._dim = getattr(config, "dimension", 512)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_image(self, image: Any) -> np.ndarray:
        """Embed a PIL.Image or raw bytes into 512-dim CLIP space (L2-normalized)."""
        pil_image = self._to_pil(image)
        t0 = time.perf_counter()
        try:
            clip_model, clip_proc = self._mm.get_clip_model()
            inputs = clip_proc(images=pil_image, return_tensors="pt").to(clip_model.device)
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
            embedding = self._l2_normalize(features.squeeze().cpu().numpy().astype(np.float32))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"CLIP image embedding failed: {exc}") from exc

        logger.debug("Image embedded in %.3fs (CLIP)", time.perf_counter() - t0)
        self._validate_dim(embedding)
        return embedding

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text into 512-dim CLIP space for caption alignment / text-to-image search."""
        if not text or not text.strip():
            text = "[empty caption]"

        t0 = time.perf_counter()
        try:
            clip_model, clip_proc = self._mm.get_clip_model()
            inputs = clip_proc(
                text=[text], return_tensors="pt", padding=True, truncation=True,
            ).to(clip_model.device)
            with torch.no_grad():
                features = clip_model.get_text_features(**inputs)
            embedding = self._l2_normalize(features.squeeze().cpu().numpy().astype(np.float32))
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"CLIP text embedding failed: {exc}") from exc

        logger.debug("Text embedded in %.3fs (CLIP)", time.perf_counter() - t0)
        self._validate_dim(embedding)
        return embedding

    # ── Helpers ───────────────────────────────────────────────

    def _to_pil(self, image: Any):
        """Convert bytes or PIL.Image to PIL.Image.Image (RGB)."""
        from PIL import Image
        try:
            if isinstance(image, bytes):
                return Image.open(io.BytesIO(image)).convert("RGB")
            if hasattr(image, "mode"):
                return image.convert("RGB")
            raise EmbeddingError(f"Unsupported image type: {type(image).__name__}")
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Could not open image: {exc}") from exc

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalize a vector. Handles near-zero norm gracefully."""
        norm = np.linalg.norm(vector)
        return vector / norm if norm >= 1e-8 else np.zeros_like(vector)

    def _validate_dim(self, embedding: np.ndarray) -> None:
        if embedding.shape[-1] != self._dim:
            raise EmbeddingError(
                f"CLIP dimension mismatch: expected {self._dim}, got {embedding.shape[-1]}"
            )
