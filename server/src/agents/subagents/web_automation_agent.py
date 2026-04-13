import os
from google.adk.agents import LlmAgent
from google.genai import types as genai_types

from ..prompts.web_automation_prompt import WEB_AUTOMATION_PROMPT
from ..tools.web_search import web_search
from ..tools.web_scraper import extract_web_content, deep_web_scrape, scrape_page

# ── Memory / context management callbacks ───────────────────────────────────
from ..memory import (
    after_model_callback,
    after_subagent_callback,
    before_tool_callback,
    after_tool_callback,
)

web_automation_agent = LlmAgent(
    name="web_automation_agent",
    model=os.getenv("REASONING_MODEL", "ollama/deepseek-v3.1:671b-cloud"),
    instruction=WEB_AUTOMATION_PROMPT,
    description="A premium web automation agent. Uses deep scraping to find reviews, bookings, and valuable offers by navigating through links logically derived from web searches.",
    generate_content_config=genai_types.GenerateContentConfig(
        tool_config=genai_types.ToolConfig(
            function_calling_config=genai_types.FunctionCallingConfig(
                mode=genai_types.FunctionCallingConfigMode.AUTO,
            ),
        ),
        temperature=0.5,
    ),
    after_model_callback=after_model_callback,
    after_agent_callback=after_subagent_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    tools=[
        web_search,           # Initial entry points
        scrape_page,          # FAST: fetch page as markdown (no LLM extraction)
        extract_web_content,  # Fast single-page extract (with LLM)
        deep_web_scrape,      # Advanced multi-page adaptive scrape
    ],
)
