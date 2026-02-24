"""OpenStreetMap / OSRM connector.

Provides geocoding (Nominatim) and routing (OSRM public demo).
"""

from __future__ import annotations

import httpx

_NOMINATIM = "https://nominatim.openstreetmap.org"
_OSRM = "https://router.project-osrm.org"
_TIMEOUT = 15
_HEADERS = {"User-Agent": "TripPlannerAgent/1.0"}


def geocode(place_name: str) -> dict:
    """Geocode a place name to coordinates using OpenStreetMap Nominatim.

    Args:
        place_name: human-readable place name, e.g. "Marina Beach, Chennai".

    Returns:
        Dict with lat, lng, display_name, or error.
    """
    try:
        resp = httpx.get(
            f"{_NOMINATIM}/search",
            headers=_HEADERS,
            params={"q": place_name, "format": "json", "limit": 1},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return {"error": f"Could not geocode '{place_name}'."}
        r = data[0]
        return {
            "lat": float(r["lat"]),
            "lng": float(r["lon"]),
            "display_name": r.get("display_name", ""),
        }
    except Exception as exc:
        return {"error": f"Geocode failed: {exc}"}


def get_route(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    mode: str = "driving",
) -> dict:
    """Get a route between two points using OSRM.

    Args:
        origin_lat: origin latitude.
        origin_lng: origin longitude.
        dest_lat: destination latitude.
        dest_lng: destination longitude.
        mode: travel mode — "driving", "walking", or "cycling".

    Returns:
        Dict with distance_km, duration_min, and route summary.
    """
    profile_map = {"driving": "car", "walking": "foot", "cycling": "bike"}
    profile = profile_map.get(mode, "car")

    try:
        coords = f"{origin_lng},{origin_lat};{dest_lng},{dest_lat}"
        resp = httpx.get(
            f"{_OSRM}/route/v1/{profile}/{coords}",
            params={"overview": "false", "steps": "false"},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok" or not data.get("routes"):
            return {"error": "No route found."}

        route = data["routes"][0]
        return {
            "distance_km": round(route["distance"] / 1000, 2),
            "duration_min": round(route["duration"] / 60, 1),
            "mode": mode,
        }
    except Exception as exc:
        return {"error": f"Routing failed: {exc}"}
