"""
routes/images.py — Image search endpoint.

GET /api/images/search?q=...&max=3  → search images via DuckDuckGo
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

router = APIRouter(tags=["images"])


@router.get("/images/search")
async def search_images(
    q: str = Query(..., description="Search query for images"),
    max: int = Query(3, ge=1, le=6, description="Max results"),
):
    """Proxy to the DuckDuckGo image search tool."""
    from src.agents.tools.image_search import search_place_images

    # Run the sync function in a thread to avoid blocking
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, lambda: search_place_images(query=q, max_results=max)
    )
    return {"images": results}
