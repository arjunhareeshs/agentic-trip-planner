"""Search Agent — AgentTool wrapper.

A specialised LlmAgent that queries the knowledge graph and ADK's built-in
Google Search to find destinations matching user preferences.  Wrapped as
AgentTool so sub-agents can invoke it like a tool.
"""

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool, google_search
import os

from agents.tools.knowledge_graph import (
    match_destinations,
    get_destination_details,
    list_all_destinations,
)

# ── Inner agent ─────────────────────────────────────────────────────────────
_search_agent_impl = LlmAgent(
    name="search_agent",
    model=os.getenv("DEFAULT_MODEL", "qwen3-vl:235b-cloud"),
    instruction="""You are a destination search specialist.

Your job is to take preference keywords and find the best matching destinations.

When given keywords:
1. Use match_destinations() to query the knowledge graph.
2. If results are sparse, broaden the keywords (e.g., "beach" → also try "coastal", "sunny").
3. Use get_destination_details() to get full info on promising matches.
4. Use google_search only when the knowledge graph lacks information
   (e.g., latest travel advisories, new attractions, current prices).
5. Use list_all_destinations() if you need to see everything available.

Return results sorted by match score with:
- Destination name & score
- Why it matched (which keywords)
- Key highlights
- Budget tier and best travel months
""",
    tools=[match_destinations, get_destination_details, list_all_destinations, google_search],
)

# ── Exported AgentTool ──────────────────────────────────────────────────────
search_agent_tool = AgentTool(agent=_search_agent_impl)
