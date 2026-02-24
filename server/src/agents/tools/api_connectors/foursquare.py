"""Foursquare Places API connector.

Venue discovery and details via Foursquare v3 API.
"""

from __future__ import annotations

import os
import httpx

_BASE = "https://api.foursquare.com/v3/places"
_TIMEOUT = 15
_HEADERS = {"Accept": "application/json"}


def _auth_headers() -> dict:
    api_key = os.getenv("FOURSQUARE_API_KEY", "")
    return {**_HEADERS, "Authorization": api_key}


def search_venues(
    query: str,
    lat: float,
    lng: float,
    radius_m: int = 5000,
    limit: int = 10,
) -> list[dict]:
    """Search Foursquare for venues near a location.

    Args:
        query: search text, e.g. "restaurant vegetarian".
        lat: latitude.
        lng: longitude.
        radius_m: radius in metres.
        limit: max results (up to 50).

    Returns:
        List of venue dicts with name, address, category, distance.
    """
    if not os.getenv("FOURSQUARE_API_KEY", ""):
        return [{"error": "FOURSQUARE_API_KEY not configured in .env"}]

    try:
        resp = httpx.get(
            f"{_BASE}/search",
            headers=_auth_headers(),
            params={
                "query": query,
                "ll": f"{lat},{lng}",
                "radius": radius_m,
                "limit": limit,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        venues = []
        for v in data.get("results", []):
            cats = [c["name"] for c in v.get("categories", [])]
            loc = v.get("location", {})
            venues.append(
                {
                    "fsq_id": v.get("fsq_id"),
                    "name": v.get("name"),
                    "categories": cats,
                    "address": loc.get("formatted_address", loc.get("address", "")),
                    "distance_m": v.get("distance"),
                    "location": {"lat": loc.get("latitude"), "lng": loc.get("longitude")},
                }
            )
        return venues if venues else [{"info": f"No venues found for '{query}' near ({lat}, {lng})."}]
    except Exception as exc:
        return [{"error": f"Foursquare search failed: {exc}"}]


def get_venue_details(fsq_id: str) -> dict:
    """Get detailed info for a Foursquare venue.

    Args:
        fsq_id: the Foursquare venue ID.

    Returns:
        Dict with name, rating, hours, price, tips, photos.
    """
    if not os.getenv("FOURSQUARE_API_KEY", ""):
        return {"error": "FOURSQUARE_API_KEY not configured in .env"}

    try:
        resp = httpx.get(
            f"{_BASE}/{fsq_id}",
            headers=_auth_headers(),
            params={"fields": "name,rating,hours,price,tips,photos,location,categories,description"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        v = resp.json()

        tips = [
            {"text": t.get("text", "")[:300], "created_at": t.get("created_at")}
            for t in v.get("tips", [])[:5]
        ]
        return {
            "fsq_id": fsq_id,
            "name": v.get("name"),
            "rating": v.get("rating"),
            "price": v.get("price"),
            "hours": v.get("hours"),
            "description": v.get("description"),
            "tips": tips,
            "categories": [c["name"] for c in v.get("categories", [])],
        }
    except Exception as exc:
        return {"error": f"Foursquare details failed: {exc}"}
