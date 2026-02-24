"""
metadata_extractor.py — Keyword-based metadata extraction from text and captions.

Detects structured attributes from text content without LLM dependency:
  • city
  • scene_type (lake, mountain, market, temple, beach, forest, desert, city)
  • crowd_level (low, medium, high)
  • lighting (sunrise, sunset, night, daylight)
  • emotion_tags (calm, romantic, adventurous, spiritual, joyful, family)

DESIGN:
  • Pure keyword/regex matching — deterministic, fast, no network calls.
  • Keyword dictionaries loaded from MetadataConfig (sourced from YAML).
  • Returns empty dict on no matches — NEVER raises.

ISOLATION: Imports ONLY from types, utils. No other RAG modules.

USAGE:
    extractor = MetadataExtractor(metadata_config)
    meta = extractor.extract("Sunset over Udaipur lake with heritage palace")
    # {'city': 'udaipur', 'scene_type': 'lake', 'lighting': 'sunset'}
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MetadataExtractor:
    """
    Keyword-based metadata extractor for text and image captions.

    Args:
        metadata_config: The metadata sub-section of RAGSettings.
                         Contains keyword dictionaries from default.yaml.
    """

    def __init__(self, metadata_config=None):
        self._config = metadata_config
        # Pre-compile city name patterns for efficiency
        self._city_patterns = self._build_city_patterns()
        logger.debug("MetadataExtractor initialized with %d city patterns", len(self._city_patterns))

    # ── Public API ────────────────────────────────────────────

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata attributes from a text string.

        Args:
            text: Any text — caption, paragraph, query string.

        Returns:
            Dict with keys: city, scene_type, crowd_level, lighting, emotion_tags.
            Only non-empty values are included. Returns {} on no matches.
            NEVER raises.
        """
        if not text or not isinstance(text, str):
            return {}

        text_lower = text.lower()
        result: Dict[str, Any] = {}

        try:
            city = self._detect_city(text_lower)
            if city:
                result["city"] = city

            scene = self._detect_scene_type(text_lower)
            if scene:
                result["scene_type"] = scene

            crowd = self._detect_crowd_level(text_lower)
            if crowd:
                result["crowd_level"] = crowd

            lighting = self._detect_lighting(text_lower)
            if lighting:
                result["lighting"] = lighting

            emotions = self._detect_emotions(text_lower)
            if emotions:
                result["emotion_tags"] = emotions

        except Exception as exc:
            logger.warning("MetadataExtractor.extract() failed gracefully: %s", exc)

        return result

    def extract_from_query(self, query: str) -> Dict[str, Any]:
        """
        Alias for extract() for clarity in query processing context.
        Returns same structure as extract().
        """
        return self.extract(query)

    # ── Detection methods ─────────────────────────────────────

    def _detect_city(self, text_lower: str) -> str:
        """Return first matching city name, or empty string."""
        for city, pattern in self._city_patterns:
            if pattern.search(text_lower):
                return city
        return ""

    def _detect_scene_type(self, text_lower: str) -> str:
        """Return first matching scene type, or empty string."""
        if not self._config or not hasattr(self._config, "scene_types"):
            return ""

        for scene_type, keywords in self._config.scene_types.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return scene_type
        return ""

    def _detect_crowd_level(self, text_lower: str) -> str:
        """Return crowd level: low | medium | high | ''."""
        if not self._config or not hasattr(self._config, "crowd_level_keywords"):
            return ""

        for level, keywords in self._config.crowd_level_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return level
        return ""

    def _detect_lighting(self, text_lower: str) -> str:
        """Return lighting condition: sunrise | sunset | night | daylight | ''."""
        if not self._config or not hasattr(self._config, "lighting_keywords"):
            return ""

        for lighting, keywords in self._config.lighting_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return lighting
        return ""

    def _detect_emotions(self, text_lower: str) -> List[str]:
        """Return list of matched emotion tags (may be empty)."""
        if not self._config or not hasattr(self._config, "emotion_keywords"):
            return []

        matched = []
        for emotion, keywords in self._config.emotion_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if emotion not in matched:
                        matched.append(emotion)
                    break  # one match per emotion is enough
        return matched

    # ── Helpers ───────────────────────────────────────────────

    def _build_city_patterns(self) -> List[tuple]:
        """
        Pre-compile regex patterns for city detection.
        Uses word boundaries to avoid false matches (e.g. "bali" in "balibago").
        """
        if not self._config or not hasattr(self._config, "cities"):
            return []

        patterns = []
        for city in self._config.cities:
            pattern = re.compile(r"\b" + re.escape(city.lower()) + r"\b")
            patterns.append((city, pattern))
        return patterns
