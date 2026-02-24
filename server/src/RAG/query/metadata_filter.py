"""
metadata_filter.py — Build LlamaIndex MetadataFilters from extracted attributes.

Implements FilterProtocol.
Takes a dictionary of attributes (city, scene_type, etc.) and converts them 
into the structured MetadataFilters object used by LlamaIndex VectorStores.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..protocols import FilterProtocol
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetadataFilterBuilder(FilterProtocol):
    """
    Builder for LlamaIndex MetadataFilters.
    """

    def build_filters(self, metadata: Dict[str, Any]) -> Optional[Any]:
        """
        Convert raw metadata dict into LlamaIndex MetadataFilters.

        Returns:
            MetadataFilters instance or None if no filters to apply.
        """
        if not metadata:
            return None

        try:
            from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

            filter_list = []
            
            # City filter
            if metadata.get("city"):
                filter_list.append(
                    MetadataFilter(key="city", value=metadata["city"])
                )
            
            # Scene type filter
            if metadata.get("scene_type"):
                filter_list.append(
                    MetadataFilter(key="scene_type", value=metadata["scene_type"])
                )

            # Crowd level filter
            if metadata.get("crowd_level"):
                filter_list.append(
                    MetadataFilter(key="crowd_level", value=metadata["crowd_level"])
                )

            if not filter_list:
                return None

            return MetadataFilters(filters=filter_list)

        except Exception as exc:
            logger.warning("Could not build metadata filter: %s", exc)
            return None
