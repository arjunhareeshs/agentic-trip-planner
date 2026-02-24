"""Itinerary Sub-Agent.

Builds detailed day-by-day itineraries using maps, places, weather, and routing.
Called by the orchestrator AFTER a destination and all preferences are confirmed.
"""

from google.adk.agents import LlmAgent  # type: ignore[import-untyped]
import os

from ..prompts.itinerary_prompt import ITINERARY_INSTRUCTION
from ..tools.api_connectors.geoapify import search_places, get_place_details, geocode, get_route
from ..tools.api_connectors.openweather import get_weather_forecast
from ..tools.web_search import web_search
from ..memory import after_subagent_callback, before_tool_callback, after_tool_callback

itinerary_agent = LlmAgent(
    name="itinerary_agent",
    model=os.getenv("REASONING_MODEL", "ollama/deepseek-v3.1:671b-cloud"),
    instruction=ITINERARY_INSTRUCTION,
    description=(
        "ONLY call this agent when the user has CONFIRMED a specific destination, "
        "duration, budget, dietary preferences, and travel mode. It builds a full "
        "day-by-day itinerary with hotels, restaurants, attractions, routes, weather, "
        "and cost breakdown. Send it all confirmed details. It does NOT talk to the user directly."
    ),
    tools=[
        search_places,
        get_place_details,
        geocode,
        get_route,
        get_weather_forecast,
        web_search,
    ],
    after_agent_callback=after_subagent_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    output_key="itinerary_result",
)
