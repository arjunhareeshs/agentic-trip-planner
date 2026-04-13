"""Maps & Places Agent — AgentTool wrapper.

A specialised LlmAgent that handles venue discovery (Foursquare), routing
(OpenStreetMap), weather (OpenWeather), and Google Search for supplementary
info.  Wrapped as AgentTool for the itinerary sub-agent.
"""

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, google_search
import os

from agents.tools.api_connectors.foursquare import (
    search_venues,
    get_venue_details,
)
from agents.tools.api_connectors.openstreetmap import geocode, get_route
from agents.tools.api_connectors.openweather import get_weather_forecast

# ── Inner agent ─────────────────────────────────────────────────────────────
_maps_places_impl = LlmAgent(
    name="maps_and_places_agent",
    model=os.getenv("DEFAULT_MODEL", "qwen3-vl:235b-cloud"),
    instruction="""You are a maps and places specialist.

Your job is to find venues, get details/reviews, calculate routes, and check weather.

When asked:
1. **Find places**: Use search_venues() with appropriate queries and coordinates
   (e.g., query="budget hotels", lat=13.05, lng=80.28).
2. **Get details & reviews**: Use get_venue_details() with the fsq_id for
   in-depth info including tips/reviews, ratings, hours, and pricing.
3. **Calculate routes**: Use geocode() to get coordinates for a place name,
   then get_route() to find distances and travel times between locations.
4. **Check weather**: Use get_weather_forecast() with destination lat/lng.
5. **Supplementary info**: Use google_search for anything the other tools
   don't cover — opening hours, entry fees, booking links, current offers.

Always return structured data with:
- Place name, rating, price range
- Address and distance from previous location
- Travel time and mode between locations
- Weather conditions for the requested dates

Be thorough — search for alternatives if the first result isn't ideal.
""",
    tools=[
        search_venues,
        get_venue_details,
        geocode,
        get_route,
        get_weather_forecast,
        google_search,
    ],
)

# ── Exported AgentTool ──────────────────────────────────────────────────────
maps_and_places_agent_tool = AgentTool(agent=_maps_places_impl)
