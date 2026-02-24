"""
Knowledge Graph tool for the agent system.

Loads the real Tourism Knowledge Graph from graph_state.json (built by the
KG pipeline in server/src/knowledgegraph/) and provides graph-traversal
tools for destination matching, filtering, and lookup.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

# ── Locate graph state file ─────────────────────────────────────────────────
_KG_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "kg_output")
_STATE_FILE = os.path.join(_KG_OUTPUT_DIR, "graph_state.json")

# ── Load graph once at import time ──────────────────────────────────────────
_NODES: dict[str, dict] = {}       # node_name → {node_type, ...properties}
_TRIPLES: list[dict] = []          # [{subject, predicate, object, subject_type, object_type}]
_DESTINATIONS: list[str] = []      # List of Destination node names
_KEYWORD_INDEX: dict[str, list[str]] = {}  # keyword → [destination_names]

# Reverse indexes for fast traversal
_DEST_EMOTIONS: dict[str, list[str]] = {}     # destination → [emotions]
_DEST_TYPES: dict[str, list[str]] = {}        # destination → [place_types]
_DEST_SEASONS: dict[str, list[str]] = {}      # destination → [seasons]
_DEST_LOCATIONS: dict[str, dict] = {}         # destination → {city, state, country, continent}
_EMOTION_DESTS: dict[str, list[str]] = {}     # emotion → [destinations]
_TYPE_DESTS: dict[str, list[str]] = {}        # place_type → [destinations]
_SEASON_DESTS: dict[str, list[str]] = {}      # season → [destinations]
_CITY_DESTS: dict[str, list[str]] = {}        # city → [destinations]
_STATE_DESTS: dict[str, list[str]] = {}       # state → [destinations]
_COUNTRY_DESTS: dict[str, list[str]] = {}     # country → [destinations]

_GRAPH_LOADED = False


def _load_graph():
    """Load graph state and build all indexes."""
    global _NODES, _TRIPLES, _DESTINATIONS, _KEYWORD_INDEX, _GRAPH_LOADED
    global _DEST_EMOTIONS, _DEST_TYPES, _DEST_SEASONS, _DEST_LOCATIONS
    global _EMOTION_DESTS, _TYPE_DESTS, _SEASON_DESTS
    global _CITY_DESTS, _STATE_DESTS, _COUNTRY_DESTS

    if not os.path.exists(_STATE_FILE):
        print(f"  ⚠️ Knowledge graph state not found at {_STATE_FILE}")
        _GRAPH_LOADED = True
        return

    with open(_STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    _NODES = state.get("nodes", {})
    _TRIPLES = state.get("triples", [])

    # Build destination list
    _DESTINATIONS.clear()
    for name, attrs in _NODES.items():
        if attrs.get("node_type") == "Destination":
            _DESTINATIONS.append(name)

    # Build relationship indexes from triples
    for triple in _TRIPLES:
        subj = triple["subject"]
        pred = triple["predicate"]
        obj = triple["object"]
        subj_type = triple.get("subject_type", "")
        obj_type = triple.get("object_type", "")

        if subj_type == "Destination":
            if pred == "EVOKES":
                _DEST_EMOTIONS.setdefault(subj, []).append(obj)
                _EMOTION_DESTS.setdefault(obj.lower(), []).append(subj)
            elif pred == "HAS_TYPE":
                _DEST_TYPES.setdefault(subj, []).append(obj)
                _TYPE_DESTS.setdefault(obj.lower(), []).append(subj)
            elif pred == "BEST_VISITED_IN":
                _DEST_SEASONS.setdefault(subj, []).append(obj)
                _SEASON_DESTS.setdefault(obj.lower(), []).append(subj)
            elif pred == "LOCATED_IN":
                loc = _DEST_LOCATIONS.setdefault(subj, {})
                if obj_type == "City":
                    loc["city"] = obj
                    _CITY_DESTS.setdefault(obj.lower(), []).append(subj)
                elif obj_type == "State":
                    loc["state"] = obj
                    _STATE_DESTS.setdefault(obj.lower(), []).append(subj)
                elif obj_type == "Country":
                    loc["country"] = obj
                    _COUNTRY_DESTS.setdefault(obj.lower(), []).append(subj)

        # Also track City → State → Country → Continent chains
        if pred == "BELONGS_TO" and subj_type == "City":
            # Find all destinations in this city and set their state
            for dest in _CITY_DESTS.get(subj.lower(), []):
                _DEST_LOCATIONS.setdefault(dest, {})["state"] = obj
                _STATE_DESTS.setdefault(obj.lower(), []).append(dest)
        if pred == "PART_OF" and subj_type == "State":
            for dest in _STATE_DESTS.get(subj.lower(), []):
                _DEST_LOCATIONS.setdefault(dest, {})["country"] = obj
                _COUNTRY_DESTS.setdefault(obj.lower(), []).append(dest)
        if pred == "ON_CONTINENT" and subj_type == "Country":
            for dest in _COUNTRY_DESTS.get(subj.lower(), []):
                _DEST_LOCATIONS.setdefault(dest, {})["continent"] = obj

    # Build keyword index from node properties + connected nodes
    _KEYWORD_INDEX.clear()
    for dest_name in _DESTINATIONS:
        keywords = set()
        # Add name words as keywords
        for word in dest_name.lower().split():
            if len(word) > 2:
                keywords.add(word)

        # Add emotions
        for emo in _DEST_EMOTIONS.get(dest_name, []):
            keywords.add(emo.lower())

        # Add types
        for pt in _DEST_TYPES.get(dest_name, []):
            keywords.add(pt.lower())

        # Add location info
        loc = _DEST_LOCATIONS.get(dest_name, {})
        for loc_val in loc.values():
            if loc_val:
                for word in str(loc_val).lower().split():
                    if len(word) > 2:
                        keywords.add(word)

        # Add node properties as keywords
        attrs = _NODES.get(dest_name, {})
        for prop_key in ("significance", "description"):
            val = attrs.get(prop_key, "")
            if val and isinstance(val, str):
                for word in val.lower().split():
                    clean = word.strip(".,;:!?()[]{}\"'")
                    if len(clean) > 3:
                        keywords.add(clean)

        # Add season info
        for season in _DEST_SEASONS.get(dest_name, []):
            keywords.add(season.lower())

        # Register all keywords
        for kw in keywords:
            _KEYWORD_INDEX.setdefault(kw, []).append(dest_name)

    _GRAPH_LOADED = True
    print(f"  ✅ Knowledge graph loaded: {len(_DESTINATIONS)} destinations, "
          f"{len(_NODES)} nodes, {len(_TRIPLES)} triples")


def _ensure_graph_loaded() -> None:
    """Lazily load the graph on the first tool call instead of at import time."""
    if not _GRAPH_LOADED:
        _load_graph()


# Lazy load — called on first tool invocation, not at import time.


# ── Tool functions (auto-wrapped by ADK) ────────────────────────────────────

def match_destinations(keywords: list[str]) -> list[dict]:
    """Match user preference keywords against the destination knowledge graph.

    Args:
        keywords: list of preference keywords extracted from the conversation,
                  e.g. ["beach", "sunny", "romantic", "seafood"].

    Returns:
        A list of dicts sorted by relevance score (descending), max 10 results.
        Each dict has: name, score, matched_keywords, emotions, place_types,
        seasons, location, rating.
    """
    _ensure_graph_loaded()

    scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}

    normalised = [k.lower().strip() for k in keywords if k.strip()]

    # Exact keyword match
    for kw in normalised:
        for dest_name in _KEYWORD_INDEX.get(kw, []):
            scores[dest_name] = scores.get(dest_name, 0) + 1
            matched.setdefault(dest_name, []).append(kw)

    # Partial/substring match fallback
    if not scores:
        for kw in normalised:
            for idx_kw, dests in _KEYWORD_INDEX.items():
                if kw in idx_kw or idx_kw in kw:
                    for dest_name in dests:
                        scores[dest_name] = scores.get(dest_name, 0) + 0.5
                        matched.setdefault(dest_name, []).append(kw)

    if not scores:
        return []

    max_score = max(scores.values())
    results = []
    for dest_name in sorted(scores, key=scores.get, reverse=True)[:10]:
        attrs = _NODES.get(dest_name, {})
        loc = _DEST_LOCATIONS.get(dest_name, {})
        results.append({
            "name": dest_name,
            "score": round(scores[dest_name] / max_score, 2),
            "matched_keywords": list(set(matched.get(dest_name, []))),
            "emotions": _DEST_EMOTIONS.get(dest_name, []),
            "place_types": _DEST_TYPES.get(dest_name, []),
            "seasons": _DEST_SEASONS.get(dest_name, []),
            "location": loc,
            "rating": attrs.get("rating"),
            "data_quality": attrs.get("data_quality", "unknown"),
        })

    return results


def get_destination_details(destination_name: str) -> dict:
    """Return full details for a single destination from the knowledge graph.

    Args:
        destination_name: the destination name, e.g. "Taj Mahal", "Marina Beach".

    Returns:
        Full destination dict with properties, emotions, types, seasons, location,
        or an error dict if not found.
    """
    _ensure_graph_loaded()

    # Try exact match first
    if destination_name in _NODES:
        attrs = _NODES[destination_name]
        if attrs.get("node_type") == "Destination":
            return _build_dest_detail(destination_name, attrs)

    # Case-insensitive search
    name_lower = destination_name.lower().strip()
    for node_name, attrs in _NODES.items():
        if node_name.lower() == name_lower and attrs.get("node_type") == "Destination":
            return _build_dest_detail(node_name, attrs)

    # Partial match
    for node_name, attrs in _NODES.items():
        if attrs.get("node_type") == "Destination" and name_lower in node_name.lower():
            return _build_dest_detail(node_name, attrs)

    return {"error": f"Destination '{destination_name}' not found in knowledge graph."}


def _build_dest_detail(name: str, attrs: dict) -> dict:
    """Build full detail dict for a destination."""
    loc = _DEST_LOCATIONS.get(name, {})
    return {
        "name": name,
        "node_type": attrs.get("node_type"),
        "rating": attrs.get("rating"),
        "data_quality": attrs.get("data_quality"),
        "description": attrs.get("description", ""),
        "significance": attrs.get("significance", ""),
        "entrance_fee": attrs.get("entrance_fee"),
        "entrance_fee_currency": attrs.get("entrance_fee_currency"),
        "visit_time_hrs": attrs.get("visit_time_hrs"),
        "avg_cost_usd": attrs.get("avg_cost_usd"),
        "emotions": _DEST_EMOTIONS.get(name, []),
        "place_types": _DEST_TYPES.get(name, []),
        "seasons": _DEST_SEASONS.get(name, []),
        "city": loc.get("city", ""),
        "state": loc.get("state", ""),
        "country": loc.get("country", ""),
        "continent": loc.get("continent", ""),
    }


def list_all_destinations() -> list[dict]:
    """List all destinations in the knowledge graph with basic info.

    Returns:
        List of dicts with name, place_types, emotions, location info, rating.
    """
    _ensure_graph_loaded()

    results = []
    for dest_name in _DESTINATIONS:
        attrs = _NODES.get(dest_name, {})
        loc = _DEST_LOCATIONS.get(dest_name, {})
        results.append({
            "name": dest_name,
            "place_types": _DEST_TYPES.get(dest_name, []),
            "emotions": _DEST_EMOTIONS.get(dest_name, []),
            "city": loc.get("city", ""),
            "state": loc.get("state", ""),
            "country": loc.get("country", ""),
            "rating": attrs.get("rating"),
        })

    return results


def filter_destinations(
    emotion: Optional[str] = None,
    place_type: Optional[str] = None,
    season: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    country: Optional[str] = None,
) -> list[dict]:
    """Filter destinations by multiple criteria using graph traversal.

    All provided criteria are AND-ed together. Destinations must match ALL
    non-None criteria to be included.

    Args:
        emotion: Filter by emotion (e.g. "romantic", "thrilling", "peaceful").
        place_type: Filter by place type (e.g. "Beach", "Temple", "Fort").
        season: Filter by best season (e.g. "Winter", "Summer", "Year-round").
        city: Filter by city name.
        state: Filter by state name.
        country: Filter by country name.

    Returns:
        Filtered list of destinations with scores based on match strength.
    """
    _ensure_graph_loaded()

    # Start with all destinations, then intersect
    candidates = set(_DESTINATIONS)

    if emotion:
        emo_key = emotion.lower().strip()
        emo_dests = set(_EMOTION_DESTS.get(emo_key, []))
        candidates &= emo_dests

    if place_type:
        type_key = place_type.lower().strip()
        type_dests = set(_TYPE_DESTS.get(type_key, []))
        # Also try partial match
        if not type_dests:
            for tk, dests in _TYPE_DESTS.items():
                if type_key in tk or tk in type_key:
                    type_dests.update(dests)
        candidates &= type_dests

    if season:
        season_key = season.lower().strip()
        season_dests = set(_SEASON_DESTS.get(season_key, []))
        if not season_dests:
            for sk, dests in _SEASON_DESTS.items():
                if season_key in sk or sk in season_key:
                    season_dests.update(dests)
        candidates &= season_dests

    if city:
        city_key = city.lower().strip()
        city_dests_set = set(_CITY_DESTS.get(city_key, []))
        if not city_dests_set:
            for ck, dests in _CITY_DESTS.items():
                if city_key in ck or ck in city_key:
                    city_dests_set.update(dests)
        candidates &= city_dests_set

    if state:
        state_key = state.lower().strip()
        state_dests_set = set(_STATE_DESTS.get(state_key, []))
        if not state_dests_set:
            for sk, dests in _STATE_DESTS.items():
                if state_key in sk or sk in state_key:
                    state_dests_set.update(dests)
        candidates &= state_dests_set

    if country:
        country_key = country.lower().strip()
        country_dests_set = set(_COUNTRY_DESTS.get(country_key, []))
        if not country_dests_set:
            for ck, dests in _COUNTRY_DESTS.items():
                if country_key in ck or ck in country_key:
                    country_dests_set.update(dests)
        candidates &= country_dests_set

    # Build results sorted by rating (descending)
    results = []
    for dest_name in candidates:
        attrs = _NODES.get(dest_name, {})
        loc = _DEST_LOCATIONS.get(dest_name, {})
        rating = attrs.get("rating")
        try:
            rating_val = float(rating) if rating else 0
        except (ValueError, TypeError):
            rating_val = 0

        results.append({
            "name": dest_name,
            "rating": rating,
            "emotions": _DEST_EMOTIONS.get(dest_name, []),
            "place_types": _DEST_TYPES.get(dest_name, []),
            "seasons": _DEST_SEASONS.get(dest_name, []),
            "city": loc.get("city", ""),
            "state": loc.get("state", ""),
            "country": loc.get("country", ""),
            "data_quality": attrs.get("data_quality", "unknown"),
            "_sort_rating": rating_val,
        })

    results.sort(key=lambda x: x["_sort_rating"], reverse=True)
    # Remove sort key from output
    for r in results:
        r.pop("_sort_rating", None)

    return results[:20]  # Max 20 results


def get_graph_stats() -> dict:
    """Return knowledge graph statistics for agent awareness.

    Returns:
        Dict with total_destinations, total_nodes, total_triples,
        nodes_by_type, available_emotions, available_place_types,
        available_seasons, available_countries.
    """
    _ensure_graph_loaded()

    nodes_by_type = {}
    for attrs in _NODES.values():
        nt = attrs.get("node_type", "Unknown")
        nodes_by_type[nt] = nodes_by_type.get(nt, 0) + 1

    return {
        "total_destinations": len(_DESTINATIONS),
        "total_nodes": len(_NODES),
        "total_triples": len(_TRIPLES),
        "nodes_by_type": nodes_by_type,
        "available_emotions": sorted(_EMOTION_DESTS.keys()),
        "available_place_types": sorted(_TYPE_DESTS.keys()),
        "available_seasons": sorted(_SEASON_DESTS.keys()),
        "available_countries": sorted(_COUNTRY_DESTS.keys()),
    }
