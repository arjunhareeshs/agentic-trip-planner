"""
settings.py — Pydantic configuration loader for the RAG pipeline.

FEATURES:
  • Loads default.yaml at module import (once)
  • Validates ALL fields with Pydantic — config errors caught at startup
  • Supports environment-variable overrides: RAG_DEVICE=cuda, RAG_LOG_LEVEL=DEBUG
  • Singleton pattern: get_settings() returns the same object on every call
  • Absolute paths resolved relative to this file's directory

USAGE:
    from config.settings import get_settings
    cfg = get_settings()
    print(cfg.models.text_embedding.name)
    print(cfg.retrieval.text_top_k)

DO NOT import settings directly in submodules.
Receive it via constructor injection from pipeline.py.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator


# ══════════════════════════════════════════════════════════════
#  Sub-models (nested structure mirrors default.yaml)
# ══════════════════════════════════════════════════════════════

class TextEmbeddingConfig(BaseModel):
    name: str = "BAAI/bge-base-en-v1.5"
    dimension: int = 768
    local_path: str = "./models/bge-base-en-v1.5"
    max_seq_length: int = 512
    batch_size: int = 32
    normalize: bool = True


class ImageEmbeddingConfig(BaseModel):
    name: str = "openai/clip-vit-base-patch32"
    dimension: int = 512
    local_path: str = "./models/clip-vit-b-32"
    image_size: int = 224


class CrossEncoderConfig(BaseModel):
    name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    local_path: str = "./models/cross-encoder"
    max_seq_length: int = 512


class ModelsConfig(BaseModel):
    text_embedding: TextEmbeddingConfig = Field(default_factory=TextEmbeddingConfig)
    image_embedding: ImageEmbeddingConfig = Field(default_factory=ImageEmbeddingConfig)
    cross_encoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)


class ChunkingConfig(BaseModel):
    strategy: str = "structure_aware"
    max_chunk_tokens: int = 512
    overlap_tokens: int = 50
    min_chunk_tokens: int = 30
    respect_elements: List[str] = Field(
        default=["heading", "table", "list", "image_caption", "paragraph"]
    )
    standalone_elements: List[str] = Field(
        default=["table", "list", "pricing_block", "image_caption"]
    )

    @field_validator("max_chunk_tokens")
    @classmethod
    def max_tokens_positive(cls, v):
        if v <= 0:
            raise ValueError("max_chunk_tokens must be > 0")
        return v


class RetrievalConfig(BaseModel):
    text_top_k: int = 15
    image_top_k: int = 10
    caption_top_k: int = 10
    rerank_top_k: int = 5
    context_token_budget: int = 4096

    @model_validator(mode="after")
    def rerank_k_lte_retrieval_k(self) -> "RetrievalConfig":
        min_retrieved = min(self.text_top_k, self.image_top_k, self.caption_top_k)
        if self.rerank_top_k > min_retrieved:
            # Auto-clamp rather than raising — graceful degradation
            object.__setattr__(self, "rerank_top_k", min_retrieved)
        return self


class MetadataConfig(BaseModel):
    cities: List[str] = Field(default_factory=list)
    scene_types: Dict[str, List[str]] = Field(default_factory=dict)
    crowd_level_keywords: Dict[str, List[str]] = Field(default_factory=dict)
    lighting_keywords: Dict[str, List[str]] = Field(default_factory=dict)
    emotion_keywords: Dict[str, List[str]] = Field(default_factory=dict)


class IndexingConfig(BaseModel):
    persist_dir: str = "../vectordb"
    text_index_name: str = "text_index"
    image_index_name: str = "image_index"
    caption_index_name: str = "caption_index"
    image_cache_dir: str = "../vectordb/image_cache"
    auto_persist: bool = True


class RoutingConfig(BaseModel):
    default_engine: str = "hybrid"
    use_subquestion: bool = True
    use_tree_index: bool = True
    # Use the same local model as the rest of the project; overridable via env.
    subquestion_llm: str = os.getenv("REASONING_MODEL", "ollama/deepseek-v3.1:671b-cloud")


class ParsingConfig(BaseModel):
    docling_preserve_tables: bool = True
    docling_preserve_images: bool = True
    docling_layout_analysis: bool = True
    min_image_width_px: int = 100
    min_image_height_px: int = 100
    supported_formats: List[str] = Field(default=[".pdf"])


# ══════════════════════════════════════════════════════════════
#  Root Settings Model
# ══════════════════════════════════════════════════════════════

class RAGSettings(BaseModel):
    """
    Root configuration object. All RAG modules receive this via injection.
    Never instantiate directly — use get_settings().
    """
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    device: str = "auto"
    log_level: str = "INFO"
    log_timing: bool = True

    # Absolute base directory of the RAG package (Pydantic v2 PrivateAttr)
    _base_dir: Path = PrivateAttr(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @field_validator("device")
    @classmethod
    def valid_device(cls, v: str) -> str:
        allowed = {"auto", "cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got '{v}'")
        return v

    @field_validator("log_level")
    @classmethod
    def valid_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v_upper

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a config-relative path to an absolute path."""
        return (self._base_dir / relative_path).resolve()

    @property
    def effective_device(self) -> str:
        """Resolve 'auto' to actual device string based on availability."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


# ══════════════════════════════════════════════════════════════
#  Loader (singleton with thread-safety)
# ══════════════════════════════════════════════════════════════

_settings_instance: Optional[RAGSettings] = None
_settings_lock = threading.Lock()
_YAML_PATH = Path(__file__).parent / "default.yaml"


def _load_yaml() -> dict:
    """Load and return the raw YAML config dict."""
    if not _YAML_PATH.exists():
        raise FileNotFoundError(
            f"RAG config file not found: {_YAML_PATH}. "
            "Ensure default.yaml exists in server/src/RAG/config/"
        )
    with open(_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(data: dict) -> dict:
    """
    Apply environment-variable overrides.
    Supported: RAG_DEVICE, RAG_LOG_LEVEL, RAG_LOG_TIMING
    """
    if "RAG_DEVICE" in os.environ:
        data["device"] = os.environ["RAG_DEVICE"]
    if "RAG_LOG_LEVEL" in os.environ:
        data["log_level"] = os.environ["RAG_LOG_LEVEL"]
    if "RAG_LOG_TIMING" in os.environ:
        data["log_timing"] = os.environ["RAG_LOG_TIMING"].lower() == "true"
    return data


def get_settings() -> RAGSettings:
    """
    Return the singleton RAGSettings instance.
    Thread-safe double-checked locking pattern.

    Raises:
        ConfigError: If default.yaml is missing or has invalid values.
    """
    global _settings_instance

    if _settings_instance is not None:
        return _settings_instance

    with _settings_lock:
        if _settings_instance is not None:
            return _settings_instance

        try:
            raw = _load_yaml()
            raw = _apply_env_overrides(raw)
            _settings_instance = RAGSettings(**raw)
        except FileNotFoundError as e:
            # Import here to avoid circular dependency at module level
            from ..utils.exceptions import ConfigError
            raise ConfigError(str(e))
        except Exception as e:
            from ..utils.exceptions import ConfigError
            raise ConfigError(
                f"Failed to load RAG configuration: {e}",
                context={"yaml_path": str(_YAML_PATH)},
            )

    return _settings_instance


def reset_settings() -> None:
    """
    Reset the singleton — for testing only.
    Allows tests to inject different configs.
    """
    global _settings_instance
    with _settings_lock:
        _settings_instance = None
