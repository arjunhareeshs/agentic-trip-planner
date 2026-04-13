"""Itinerary Sub-Agent.

Creates detailed day-by-day travel itineraries with venues,
routes, hotels, restaurants, costs, and weather.
"""

from google.adk.agents import LlmAgent
import os

from agents.prompts.itinerary_prompt import ITINERARY_INSTRUCTION
from agents.tools.agenttools.maps_and_places_agent import maps_and_places_agent_tool

itinerary_agent = LlmAgent(
    name="itinerary_agent",
    model=os.getenv("REASONING_MODEL", "deepseek-v3.1:671b-cloud"),
    instruction=ITINERARY_INSTRUCTION,
    description=(
        "Creates detailed day-by-day trip itineraries. Uses maps, places, weather, "
        "and routing tools to build comprehensive plans within the user's budget "
        "including hotels, restaurants, attractions, transport costs, and timings."
    ),
    tools=[maps_and_places_agent_tool],
)
