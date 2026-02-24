"""Embedding package — BGE text embedder, CLIP image embedder, model manager."""
from .model_manager import ModelManager
from .text_embedder import BGETextEmbedder
from .image_embedder import CLIPImageEmbedder

__all__ = ["ModelManager", "BGETextEmbedder", "CLIPImageEmbedder"]
