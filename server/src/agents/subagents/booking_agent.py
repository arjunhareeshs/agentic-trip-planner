"""Booking Sub-Agent.

Searches flights, hotels, and provides booking links and step-by-step guidance.
Called by the orchestrator ONLY when the user explicitly asks to book.
"""

from google.adk.agents import LlmAgent  # type: ignore[import-untyped]
import os

from ..prompts.booking_prompt import BOOKING_INSTRUCTION
from ..tools.api_connectors.aviationstack import search_flights
from ..tools.api_connectors.geoapify import geocode, search_places, get_place_details
from ..tools.web_search import web_search
from ..tools.web_scraper import extract_web_content
from ..memory import after_subagent_callback, before_tool_callback, after_tool_callback

booking_agent = LlmAgent(
    name="booking_agent",
    model=os.getenv("REASONING_MODEL", "ollama/deepseek-v3.1:671b-cloud"),
    instruction=BOOKING_INSTRUCTION,
    description=(
        "ONLY call this agent when the user explicitly asks to book flights, hotels, "
        "or wants booking links and guidance. It searches for flights, accommodation, "
        "and returns booking options with URLs and step-by-step instructions. "
        "It does NOT talk to the user directly."
    ),
    tools=[
        search_flights,    # flight search by IATA codes (worldwide)
        geocode,           # destination name → lat/lng for hotel search
        search_places,     # hotel and venue search by coordinates
        get_place_details, # full venue details
        web_search,        # booking URLs, train/bus options, visa info
        extract_web_content, # Read full booking pages, confirm prices and amenities
    ],
    after_agent_callback=after_subagent_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    output_key="booking_result",
)
