"""
agents/agent.py — ADK entry point.

Defines `root_agent`, the orchestrator LlmAgent that coordinates
all sub-agents for end-to-end trip planning.

Run with:
    cd "c:\\agentic ai\\agent-trip-planner\\server\\src"
    adk web
Then select "agents" from the dropdown.
"""

from google.adk.agents import LlmAgent  # type: ignore[import-untyped]
from google.genai import types as genai_types  # type: ignore[import-untyped]
import os

from .prompts.orchestrator_prompt import ORCHESTRATOR_INSTRUCTION
from .subagents.preference_agent import preference_agent
from .subagents.itinerary_agent import itinerary_agent
from .subagents.booking_agent import booking_agent
from .subagents.image_analysis_agent import image_analysis_agent
from .subagents.web_automation_agent import web_automation_agent

# ── Direct tools for quick lookups (orchestrator handles these itself) ──────
from .tools.api_connectors.openweather import get_weather_forecast
from .tools.api_connectors.geoapify import geocode
from .tools.knowledge_graph import (
    match_destinations,
    filter_destinations,
    get_destination_details,
    list_all_destinations,
    get_graph_stats,
)
from .tools.web_search import web_search
from .tools.image_search import search_place_images
from .tools.web_scraper import extract_web_content, scrape_page

# ── Memory / context management callbacks ───────────────────────────────────
from .memory import (
    before_model_callback,
    after_model_callback,
    after_root_agent_callback,
    before_tool_callback,
    after_tool_callback,
)

# ── Root Agent (Orchestrator) ───────────────────────────────────────────────
root_agent = LlmAgent(
    name="trip_planner_orchestrator",
    model=os.getenv("DEFAULT_MODEL", "ollama/qwen3-vl:235b-cloud"),
    instruction=ORCHESTRATOR_INSTRUCTION,
    description="Main orchestrator that talks to the user, gathers travel preferences through conversation, and delegates to sub-agents only when enough information is collected.",
    generate_content_config=genai_types.GenerateContentConfig(
        tool_config=genai_types.ToolConfig(
            function_calling_config=genai_types.FunctionCallingConfig(
                mode=genai_types.FunctionCallingConfigMode.AUTO,
            ),
        ),
        temperature=0.7,
    ),
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=after_root_agent_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    tools=[
        # Quick lookups the orchestrator handles directly
        get_weather_forecast,   # weather forecasts
        geocode,                # place name → lat/lng
        match_destinations,     # knowledge graph keyword search (600+ worldwide destinations)
        filter_destinations,    # precise multi-criteria filter by emotion/type/season/country
        get_destination_details,# full destination info
        list_all_destinations,  # browse all destinations in the knowledge graph
        get_graph_stats,        # check available emotions, types, seasons, countries in KG
        web_search,             # general web search (last resort)
        search_place_images,    # image search — show real destination photos inline
        extract_web_content,    # web scraping tool to scrape and extract struct docs via LLM
        scrape_page,             # FAST page scraper — returns markdown, no LLM extraction
    ],
    sub_agents=[
        preference_agent,       # deep destination comparison
        itinerary_agent,        # day-by-day itinerary building
        booking_agent,          # flights + hotels + booking links
        image_analysis_agent,   # VLM image analysis (LLaVA)
        web_automation_agent,   # Advanced multi-page deep scraping
    ],
)
