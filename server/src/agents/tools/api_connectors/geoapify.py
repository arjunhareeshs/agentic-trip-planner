"""Geoapify API connector.

Provides geocoding, place search, place details, and routing.
Single API key covers all endpoints — geoapify.com/api
"""

from __future__ import annotations

import httpx
import os
from utils.http_client import http_client

_BASE = "https://api.geoapify.com/v1"
_PLACES = "https://api.geoapify.com/v2/places"
_PLACE_DETAILS = "https://api.geoapify.com/v2/place-details"
_ROUTING = "https://api.geoapify.com/v1/routing"


def _api_key() -> str:
    return os.getenv("GEOAPIFY_API_KEY", "")


# ── Geocoding ───────────────────────────────────────────────────────────────

def geocode(place_name: str) -> dict:
    """Geocode a place name to coordinates using Geoapify.

    Args:
        place_name: human-readable place name, e.g. "Marina Beach, Chennai".

    Returns:
        Dict with lat, lng, display_name, or error.
    """
    key = _api_key()
    if not key:
        print("⚠️ [Geoapify] GEOAPIFY_API_KEY not configured in .env")
        return {"error": "GEOAPIFY_API_KEY not configured in .env"}

    try:
        resp = http_client.get(
            f"{_BASE}/geocode/search",
            params={"text": place_name, "apiKey": key, "limit": 1, "format": "json"}
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            return {"error": f"Could not geocode '{place_name}'."}

        r = results[0]
        return {
            "lat": r.get("lat"),
            "lng": r.get("lon"),
            "display_name": r.get("formatted", ""),
        }
    except httpx.HTTPStatusError as exc:
        print(f"❌ [Geoapify Geocode] HTTP error {exc.response.status_code}: {exc.response.text}")
        return {"error": f"Geoapify API returned {exc.response.status_code}"}
    except httpx.RequestError as exc:
        print(f"❌ [Geoapify Geocode] Request error: {exc}")
        return {"error": "Failed to connect to Geoapify API"}
    except Exception as exc:
        print(f"❌ [Geoapify Geocode] Unexpected error: {exc}")
        return {"error": f"Geocode failed: {exc}"}


# ── Place / Venue Search ────────────────────────────────────────────────────

def search_places(
    query: str,
    lat: float,
    lng: float,
    radius_m: int = 5000,
    limit: int = 10,
) -> list[dict]:
    """Search for places (hotels, restaurants, attractions) near a location.

    Args:
        query: what to search for, e.g. "hotel", "restaurant vegetarian",
               "tourist attraction", "museum".
        lat: latitude of the center point.
        lng: longitude of the center point.
        radius_m: search radius in metres (default 5000).
        limit: maximum results to return (default 10, max 20).

    Returns:
        List of place dicts with name, address, categories, distance, place_id.
    """
    key = _api_key()
    if not key:
        print("⚠️ [Geoapify] GEOAPIFY_API_KEY not configured in .env")
        return [{"error": "GEOAPIFY_API_KEY not configured in .env"}]

    # Map common queries to Geoapify categories
    category_map = {
        "hotel": "accommodation.hotel",
        "hostel": "accommodation.hostel",
        "restaurant": "catering.restaurant",
        "cafe": "catering.cafe",
        "bar": "catering.bar",
        "museum": "entertainment.museum",
        "temple": "religion.place_of_worship",
        "beach": "beach",
        "park": "leisure.park",
        "attraction": "tourism.attraction",
        "tourist": "tourism",
        "shopping": "commercial.shopping_mall",
    }

    # Try to find matching categories
    matched_cats = []
    query_lower = query.lower()
    for keyword, cat in category_map.items():
        if keyword in query_lower:
            matched_cats.append(cat)

    # If no category matched, try broad tourism + commercial + catering + accommodation
    if not matched_cats:
        matched_cats = [
            "tourism",
            "accommodation",
            "catering",
            "entertainment",
            "leisure",
        ]

    try:
        params = {
            "categories": ",".join(matched_cats),
            "conditions": "named",
            "filter": f"circle:{lng},{lat},{radius_m}",
            "bias": f"proximity:{lng},{lat}",
            "limit": min(limit, 20),
            "apiKey": key,
        }

        resp = http_client.get(_PLACES, params=params)
        resp.raise_for_status()
        data = resp.json()

        places = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            name = props.get("name", "")
            if not name:
                continue
            places.append({
                "place_id": props.get("place_id", ""),
                "name": name,
                "categories": props.get("categories", []),
                "address": props.get("formatted", props.get("address_line2", "")),
                "distance_m": props.get("distance", 0),
                "lat": props.get("lat"),
                "lng": props.get("lon"),
            })

        if not places:
            # Fallback: try text-based geocode search
            return _text_place_search(query, lat, lng, radius_m, limit, key)

        return places
    except httpx.HTTPStatusError as exc:
        print(f"❌ [Geoapify Places] HTTP error {exc.response.status_code}: {exc.response.text}")
        return [{"error": f"Geoapify API returned {exc.response.status_code}"}]
    except httpx.RequestError as exc:
        print(f"❌ [Geoapify Places] Request error: {exc}")
        return [{"error": "Failed to connect to Geoapify API"}]
    except Exception as exc:
        print(f"❌ [Geoapify Places] Unexpected error: {exc}")
        return [{"error": f"Place search failed: {exc}"}]


def _text_place_search(
    query: str, lat: float, lng: float, radius_m: int, limit: int, key: str
) -> list[dict]:
    """Fallback text-based place search using geocode autocomplete."""
    try:
        resp = http_client.get(
            f"{_BASE}/geocode/autocomplete",
            params={
                "text": query,
                "bias": f"proximity:{lng},{lat}",
                "limit": min(limit, 20),
                "apiKey": key,
                "type": "amenity",
            }
        )
        resp.raise_for_status()
        data = resp.json()

        places = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            name = props.get("name", props.get("formatted", ""))
            if not name:
                continue
            places.append({
                "place_id": props.get("place_id", ""),
                "name": name,
                "categories": [props.get("result_type", "place")],
                "address": props.get("formatted", ""),
                "distance_m": 0,
                "lat": props.get("lat"),
                "lng": props.get("lon"),
            })
        return places if places else [{"error": f"No places found for '{query}' near ({lat}, {lng})"}]
    except httpx.HTTPStatusError as exc:
        print(f"❌ [Geoapify Text Search] HTTP error {exc.response.status_code}: {exc.response.text}")
        return [{"error": f"Geoapify API returned {exc.response.status_code}"}]
    except httpx.RequestError as exc:
        print(f"❌ [Geoapify Text Search] Request error: {exc}")
        return [{"error": "Failed to connect to Geoapify API"}]
    except Exception as exc:
        print(f"❌ [Geoapify Text Search] Unexpected error: {exc}")
        return [{"error": f"Text place search failed: {exc}"}]


# ── Place Details ───────────────────────────────────────────────────────────

def get_place_details(place_id: str) -> dict:
    """Get detailed info for a place by its Geoapify place_id.

    Args:
        place_id: the Geoapify place ID from a search result.

    Returns:
        Dict with name, address, categories, website, phone, opening_hours,
        or error.
    """
    key = _api_key()
    if not key:
        print("⚠️ [Geoapify] GEOAPIFY_API_KEY not configured in .env")
        return {"error": "GEOAPIFY_API_KEY not configured in .env"}

    try:
        resp = http_client.get(
            _PLACE_DETAILS,
            params={
                "id": place_id,
                "features": "details,description",
                "apiKey": key,
            }
        )
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            return {"error": f"No details found for place_id '{place_id}'"}

        props = features[0].get("properties", {})
        return {
            "place_id": place_id,
            "name": props.get("name", "Unknown"),
            "address": props.get("formatted", ""),
            "categories": props.get("categories", []),
            "website": props.get("website", ""),
            "phone": props.get("contact", {}).get("phone", "") if isinstance(props.get("contact"), dict) else "",
            "opening_hours": props.get("opening_hours", ""),
            "description": props.get("description", ""),
        }
    except httpx.HTTPStatusError as exc:
        print(f"❌ [Geoapify Details] HTTP error {exc.response.status_code}: {exc.response.text}")
        return {"error": f"Geoapify API returned {exc.response.status_code}"}
    except httpx.RequestError as exc:
        print(f"❌ [Geoapify Details] Request error: {exc}")
        return {"error": "Failed to connect to Geoapify API"}
    except Exception as exc:
        print(f"❌ [Geoapify Details] Unexpected error: {exc}")
        return {"error": f"Place details failed: {exc}"}


# ── Routing ─────────────────────────────────────────────────────────────────

def get_route(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    mode: str = "drive",
) -> dict:
    """Get route/distance between two points using Geoapify Routing.

    Args:
        origin_lat: origin latitude.
        origin_lng: origin longitude.
        dest_lat: destination latitude.
        dest_lng: destination longitude.
        mode: travel mode — "drive", "walk", "bicycle", or "transit".

    Returns:
        Dict with distance_km, duration_min, and mode.
    """
    key = _api_key()
    if not key:
        print("⚠️ [Geoapify] GEOAPIFY_API_KEY not configured in .env")
        return {"error": "GEOAPIFY_API_KEY not configured in .env"}

    valid_modes = {"drive", "walk", "bicycle", "transit"}
    if mode not in valid_modes:
        mode = "drive"

    try:
        waypoints = f"{origin_lat},{origin_lng}|{dest_lat},{dest_lng}"
        resp = http_client.get(
            _ROUTING,
            params={
                "waypoints": waypoints,
                "mode": mode,
                "apiKey": key,
            }
        )
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            return {"error": "No route found."}

        props = features[0].get("properties", {})
        return {
            "distance_km": round(props.get("distance", 0) / 1000, 2),
            "duration_min": round(props.get("time", 0) / 60, 1),
            "mode": mode,
        }
    except httpx.HTTPStatusError as exc:
        print(f"❌ [Geoapify Routing] HTTP error {exc.response.status_code}: {exc.response.text}")
        return {"error": f"Geoapify API returned {exc.response.status_code}"}
    except httpx.RequestError as exc:
        print(f"❌ [Geoapify Routing] Request error: {exc}")
        return {"error": "Failed to connect to Geoapify API"}
    except Exception as exc:
        print(f"❌ [Geoapify Routing] Unexpected error: {exc}")
        return {"error": f"Routing failed: {exc}"}
