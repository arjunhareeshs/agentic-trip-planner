"""
routes/geocode.py — Geocoding / map endpoints.

GET /api/geocode?place=...  → geocode a place name to lat/lng
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

router = APIRouter(tags=["geocode"])


@router.get("/geocode")
async def geocode_place(
    place: str = Query(..., description="Place name to geocode"),
):
    """Geocode a place name using Geoapify."""
    from src.agents.tools.api_connectors.geoapify import geocode

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: geocode(place))
    return result
