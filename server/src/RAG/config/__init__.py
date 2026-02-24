"""Config package — exposes get_settings() singleton."""
from .settings import get_settings, RAGSettings

__all__ = ["get_settings", "RAGSettings"]
