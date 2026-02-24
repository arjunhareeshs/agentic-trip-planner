"""
model_manager.py — Thread-safe model loader for pre-downloaded local models.

Pre-downloaded models (via download_models.py):
  • BGE-base-en-v1.5  → ./models/bge-base-en-v1.5
  • CLIP ViT-B/32     → ./models/clip-vit-b-32
  • Cross-encoder     → downloaded from HuggingFace on first use

ISOLATION: Imports ONLY from utils. No other RAG modules.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from ..utils.exceptions import EmbeddingError
from ..utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()


class ModelManager:
    """
    Lazy-loading, thread-safe model manager.
    Loads BGE and CLIP from pre-downloaded local directories.
    """

    def __init__(self, models_config=None, device: str = "cpu", base_dir: Optional[Path] = None):
        self._config = models_config
        self._device = device
        self._base_dir = base_dir or Path(__file__).resolve().parent.parent
        self._text_model = None
        self._clip_model = None
        self._clip_processor = None
        self._cross_encoder = None
        logger.info("ModelManager initialized on device: %s", device)

    def _resolve_local(self, local_path: str) -> Path:
        """Resolve a config-relative local_path to absolute."""
        return (self._base_dir / local_path).resolve()

    # ── Text Embedding Model (BGE) ────────────────────────────

    def get_text_model(self):
        """Load BGE-base-en-v1.5 from local directory. Thread-safe, cached."""
        if self._text_model is not None:
            return self._text_model

        with _lock:
            if self._text_model is not None:
                return self._text_model

            local_path = getattr(
                getattr(self._config, "text_embedding", None),
                "local_path", "./models/bge-base-en-v1.5"
            )
            model_path = self._resolve_local(local_path)

            if not model_path.exists():
                raise EmbeddingError(
                    f"BGE model not found at {model_path}. Run download_models.py first.",
                    context={"path": str(model_path)},
                )

            logger.info("Loading BGE from local: %s", model_path)
            t0 = time.perf_counter()
            try:
                from sentence_transformers import SentenceTransformer

                self._text_model = SentenceTransformer(str(model_path)).to(self._device)
                logger.info("BGE loaded in %.1fs on %s", time.perf_counter() - t0, self._device)
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers",
                ) from exc
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load BGE from '{model_path}': {exc}",
                    context={"path": str(model_path), "device": self._device},
                ) from exc

        return self._text_model

    # ── Image Embedding Model (CLIP) ──────────────────────────

    def get_clip_model(self) -> Tuple[Any, Any]:
        """Load CLIP ViT-B/32 from local directory. Thread-safe, cached."""
        if self._clip_model is not None:
            return self._clip_model, self._clip_processor

        with _lock:
            if self._clip_model is not None:
                return self._clip_model, self._clip_processor

            local_path = getattr(
                getattr(self._config, "image_embedding", None),
                "local_path", "./models/clip-vit-b-32"
            )
            model_path = self._resolve_local(local_path)

            if not model_path.exists():
                raise EmbeddingError(
                    f"CLIP model not found at {model_path}. Run download_models.py first.",
                    context={"path": str(model_path)},
                )

            logger.info("Loading CLIP from local: %s", model_path)
            t0 = time.perf_counter()
            try:
                from transformers import CLIPModel, CLIPProcessor

                self._clip_model = CLIPModel.from_pretrained(
                    str(model_path), use_safetensors=True, local_files_only=True,
                ).to(self._device)
                self._clip_processor = CLIPProcessor.from_pretrained(
                    str(model_path), local_files_only=True,
                )
                logger.info("CLIP loaded in %.1fs on %s", time.perf_counter() - t0, self._device)
            except ImportError as exc:
                raise EmbeddingError(
                    "transformers not installed. Run: pip install transformers",
                ) from exc
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load CLIP from '{model_path}': {exc}",
                    context={"path": str(model_path), "device": self._device},
                ) from exc

        return self._clip_model, self._clip_processor

    # ── Cross-Encoder Model ───────────────────────────────────

    def get_cross_encoder(self):
        """Load cross-encoder reranking model. Thread-safe, cached."""
        if self._cross_encoder is not None:
            return self._cross_encoder

        with _lock:
            if self._cross_encoder is not None:
                return self._cross_encoder

            model_name = getattr(
                getattr(self._config, "cross_encoder", None),
                "name", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

            logger.info("Loading cross-encoder: %s", model_name)
            t0 = time.perf_counter()
            try:
                from sentence_transformers.cross_encoder import CrossEncoder

                self._cross_encoder = CrossEncoder(model_name, device=self._device)
                logger.info("Cross-encoder loaded in %.1fs", time.perf_counter() - t0)
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers",
                ) from exc
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load cross-encoder '{model_name}': {exc}",
                    context={"model": model_name},
                ) from exc

        return self._cross_encoder

    # ── Health check ──────────────────────────────────────────

    def health_check(self) -> dict:
        """Verify all models are functional. Returns {name: True/False}."""
        results = {"text": False, "clip": False, "cross_encoder": False}

        try:
            self.get_text_model().encode(["health check"])
            results["text"] = True
        except Exception as exc:
            logger.warning("Text model health check failed: %s", exc)

        try:
            clip_model, clip_proc = self.get_clip_model()
            inputs = clip_proc(text=["health check"], return_tensors="pt", padding=True)
            inputs = {k: v.to(clip_model.device) for k, v in inputs.items()}
            clip_model.get_text_features(**inputs)
            results["clip"] = True
        except Exception as exc:
            logger.warning("CLIP health check failed: %s", exc)

        try:
            self.get_cross_encoder().predict([("health check", "test passage")])
            results["cross_encoder"] = True
        except Exception as exc:
            logger.warning("Cross-encoder health check failed: %s", exc)

        return results
