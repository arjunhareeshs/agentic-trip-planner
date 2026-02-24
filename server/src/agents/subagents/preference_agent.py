"""Preference Sub-Agent.

Profiles the traveller against the knowledge graph (70%) and applies
LLM reasoning (30%) to return 2-4 ranked worldwide destination
recommendations with embedded photo references.
"""

from google.adk.agents import LlmAgent  # type: ignore[import-untyped]
import os

from ..prompts.preference_prompt import PREFERENCE_INSTRUCTION
from ..tools.knowledge_graph import (
    match_destinations,
    filter_destinations,
    get_destination_details,
    list_all_destinations,
    get_graph_stats,
)
from ..tools.web_search import web_search
from ..tools.image_search import search_place_images
from ..memory import after_subagent_callback, before_tool_callback, after_tool_callback

preference_agent = LlmAgent(
    name="preference_agent",
    model=os.getenv("REASONING_MODEL", "ollama/deepseek-v3.1:671b-cloud"),
    instruction=PREFERENCE_INSTRUCTION,
    description=(
        "Call this agent when you have collected the user's trip type/vibe, "
        "approximate budget, and duration. It profiles the traveller, queries "
        "the knowledge graph, applies 70/30 KG+reasoning logic, and returns "
        "2-4 ranked worldwide destinations with photos. "
        "It does NOT talk to the user directly."
    ),
    tools=[
        get_graph_stats,          # Step 1: always first - understand KG vocabulary
        filter_destinations,      # Step 2: precise multi-criteria AND-filter
        match_destinations,       # Step 3: broad semantic keyword search
        get_destination_details,  # Step 5: full profile for top 4 candidates
        list_all_destinations,    # Step 6: fallback - scan full catalog
        web_search,               # Step 6: fallback - worldwide alternatives
        search_place_images,      # Step 7: 2 photos per final destination
    ],
    after_agent_callback=after_subagent_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    output_key="preference_results",
)
