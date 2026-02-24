"""Image Analysis Sub-Agent.

Uses a Vision Language Model (LLaVA) to analyze travel images.
Identifies landmarks, destinations, vibes, and provides travel suggestions.
Called by the orchestrator when the user shares a travel photo.
"""

from google.adk.agents import LlmAgent
import os

from ..prompts.image_analysis_prompt import IMAGE_ANALYSIS_INSTRUCTION
from ..tools.web_search import web_search
from ..tools.image_search import search_place_images
from ..memory import after_subagent_callback, after_tool_callback

image_analysis_agent = LlmAgent(
    name="image_analysis_agent",
    model="ollama/llava:latest",
    instruction=IMAGE_ANALYSIS_INSTRUCTION,
    description=(
        "ONLY call this agent when the user shares an IMAGE or PHOTO and wants "
        "it analyzed. It uses a vision model to identify landmarks, destinations, "
        "vibes, and suggest similar places. Send it the image. "
        "It does NOT talk to the user directly."
    ),
    tools=[
        web_search,           # identify ambiguous landmarks, find visitor info
        search_place_images,  # fetch real photo references for identified place + similar destinations
    ],
    after_agent_callback=after_subagent_callback,
    after_tool_callback=after_tool_callback,
    output_key="image_analysis_result",
)
