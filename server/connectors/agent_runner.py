"""
connectors/agent_runner.py — Bridge between FastAPI routes and the ADK agent.

Provides async methods to:
  • Create sessions
  • Send messages to the agent and collect full or streamed responses
  • Extract images and map data from agent responses
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, AsyncGenerator

from google.adk.runners import Runner  # type: ignore[import-untyped]
from google.adk.sessions import InMemorySessionService  # type: ignore[import-untyped]
from google.genai import types as genai_types  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ── Lazy singleton ──────────────────────────────────────────────────────────
_runner: Runner | None = None
_session_service: InMemorySessionService | None = None
_APP_NAME = "trip_planner"


def _get_runner() -> Runner:
    """Create the ADK Runner singleton (lazy init)."""
    global _runner, _session_service

    if _runner is not None:
        return _runner

    # Import the agent + session service from the agents package
    from src.agents.agent import root_agent
    from src.agents.memory.services import session_service

    _session_service = session_service

    _runner = Runner(
        agent=root_agent,
        app_name=_APP_NAME,
        session_service=session_service,
    )
    logger.info("ADK Runner initialised with agent: %s", root_agent.name)
    return _runner


def _get_session_service() -> InMemorySessionService:
    """Return the session service (initialise runner first if needed)."""
    _get_runner()
    assert _session_service is not None, "Session service not initialised"
    return _session_service


# ── Image / Map extraction helpers ──────────────────────────────────────────

# Matches Markdown images: ![alt](url)
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Matches coordinate patterns: lat, lng or (lat, lng)
_COORD_RE = re.compile(
    r"(?:latitude|lat)[:\s]*(-?\d+\.?\d*)[,\s]+(?:longitude|lng|lon)[:\s]*(-?\d+\.?\d*)",
    re.IGNORECASE,
)


def _extract_images_from_text(text: str) -> list[dict]:
    """Extract image URLs from Markdown image syntax in bot response."""
    images = []
    for match in _IMG_RE.finditer(text):
        alt, url = match.group(1), match.group(2)
        if url and not url.startswith("data:"):
            images.append({
                "title": alt or "Destination image",
                "image_url": url,
                "source_url": "",
            })
    return images


def _extract_map_data(text: str) -> dict | None:
    """Try to extract lat/lng coordinates from the bot response."""
    match = _COORD_RE.search(text)
    if match:
        try:
            return {
                "lat": float(match.group(1)),
                "lng": float(match.group(2)),
            }
        except (ValueError, IndexError):
            pass
    return None


# Patterns that indicate tool call / internal content that shouldn't be shown to users
_TOOL_CALL_PATTERNS = [
    re.compile(r'^\s*```tool_code', re.MULTILINE),
    re.compile(r'^\s*```python\s*\n.*?\bdef\b', re.MULTILINE | re.DOTALL),
    re.compile(r'\{\s*"function_call"', re.IGNORECASE),
    re.compile(r'\{\s*"name"\s*:\s*"\w+"\s*,\s*"args"\s*:', re.IGNORECASE),
    re.compile(r'\{\s*"tool_name"\s*:', re.IGNORECASE),
    re.compile(r'Tool call:\s*\w+', re.IGNORECASE),
    re.compile(r'Calling tool:\s*\w+', re.IGNORECASE),
    re.compile(r'Function call:\s*\w+', re.IGNORECASE),
    re.compile(r'<function_call>', re.IGNORECASE),
    re.compile(r'<tool_call>', re.IGNORECASE),
]

# Tool / agent names that should never appear in user-visible text
_INTERNAL_NAMES = [
    "web_search", "scrape_page", "extract_web_content", "deep_web_scrape",
    "search_place_images", "get_weather_forecast", "geocode",
    "match_destinations", "filter_destinations", "get_destination_details",
    "list_all_destinations", "get_graph_stats",
    "preference_agent", "itinerary_agent", "booking_agent",
    "image_analysis_agent", "web_automation_agent",
    "trip_planner_orchestrator",
]


def _clean_tool_call_text(text: str) -> str:
    """Remove tool call parameters, function call JSON, and other internal content from text."""
    if not text or not text.strip():
        return ""

    stripped = text.strip()

    # 1) If the entire text is a JSON object with tool-call keys, skip it
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            parsed = json.loads(stripped)
            if any(k in parsed for k in ('function_call', 'name', 'tool_name', 'args', 'parameters')):
                return ""
        except (json.JSONDecodeError, TypeError):
            pass

    # 2) Check if any tool-call pattern matches the entire chunk → skip the whole thing
    for pattern in _TOOL_CALL_PATTERNS:
        if pattern.search(stripped):
            return ""

    # 3) Remove ```json ... ``` / ```tool_code ... ``` / ``` ... ``` code blocks containing tool params
    text = re.sub(
        r'```(?:json|tool_code|python|)\s*\n?\s*\{[^`]*?\}\s*\n?\s*```',
        '', text, flags=re.DOTALL | re.IGNORECASE,
    )

    # 4) Remove ```tool_code ... ``` blocks (any content)
    text = re.sub(r'```tool_code.*?```', '', text, flags=re.DOTALL)

    # 5) Remove <function_call>...</function_call> and <tool_call>...</tool_call>
    text = re.sub(r'<function_call>.*?</function_call>', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)

    # 6) Remove bare JSON blocks that look like tool calls (even without code fences)
    text = re.sub(
        r'\{\s*"(?:name|function_call|tool_name|tool_call)"\s*:.*?\}(?:\s*\})?',
        '', text, flags=re.DOTALL,
    )

    # 6b) Remove empty code block markers (leftover after removing JSON content)
    text = re.sub(r'```(?:json|tool_code|python|)\s*```', '', text, flags=re.IGNORECASE)
    # Also catch lone backtick fences with nothing between them
    text = re.sub(r'`{3,}\s*`{3,}', '', text)
    # And single lone backtick pairs with only whitespace
    text = re.sub(r'`\s*`', '', text)

    # 7) Remove lines that reference internal tool/agent names directly
    #    e.g., "Let me call web_search..." or "I'll use extract_web_content..."
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        lower = line.lower().strip()

        # Skip empty backtick-only lines
        if re.match(r'^`+$', lower):
            continue

        # Skip lines that narrate tool usage
        if any(name in lower for name in _INTERNAL_NAMES):
            narration_signals = [
                'let me', "i'll", 'i will', 'calling', 'using', 'invoke',
                'delegate', 'transfer', 'pass to', 'send to', 'call ',
                'searching', 'scraping', 'extracting', 'fetching',
                'start by', 'going to', 'need to',
            ]
            if any(sig in lower for sig in narration_signals):
                continue

        # Skip general process-narration lines (even without tool names)
        process_narration = [
            "let me search", "let me scrape", "let me extract",
            "let me look up", "let me find", "let me check",
            "i'll start by searching", "i'll search for",
            "i'll scrape", "i'll extract", "i'll look up",
            "i notice the web extraction didn't work",
            "the tool returned", "the search returned",
            "based on my search results", "from the search results",
            "here are the search results",
            "i'll use", "let me use", "i need to call",
        ]
        if any(phrase in lower for phrase in process_narration):
            continue

        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # 8) Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _extract_place_name(text: str) -> str | None:
    """Try to extract a likely destination/place name from response text."""
    # Look for patterns like "in <Place>" or "destination: <Place>"
    patterns = [
        re.compile(r"(?:destination|location|place|city|visiting|travel to|trip to|heading to)[:\s]+([A-Z][a-zA-Z\s,]+)", re.IGNORECASE),
        re.compile(r"(?:Welcome to|Explore|Discover)\s+([A-Z][a-zA-Z\s,]+)", re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1).strip().rstrip(",. ")
    return None


# ── Controller ──────────────────────────────────────────────────────────────

class AgentRunnerController:
    """High-level controller wrapping the ADK Runner for the REST API."""

    async def create_session(
        self, session_id: str, user_id: str = "default_user"
    ) -> str:
        """Create a new ADK session."""
        svc = _get_session_service()
        session = await svc.create_session(
            app_name=_APP_NAME,
            user_id=user_id,
        )
        return session.id

    async def get_session_info(self, session_id: str) -> dict:
        """Return basic session metadata."""
        svc = _get_session_service()
        try:
            session = await svc.get_session(
                app_name=_APP_NAME,
                user_id="default_user",
                session_id=session_id,
            )
            if session:
                return {
                    "session_id": session.id,
                    "user_id": session.user_id,
                    "turns": len(session.events) if hasattr(session, "events") else 0,
                    "state_keys": list(session.state.keys()) if session.state else [],
                }
        except Exception:
            pass
        return {"session_id": session_id, "error": "Session not found"}

    async def run_agent(
        self,
        message: str,
        session_id: str,
        user_id: str = "default_user",
    ) -> dict:
        """Send a message and collect the full response (non-streaming)."""
        runner = _get_runner()

        # Create or get session
        svc = _get_session_service()
        try:
            session = await svc.get_session(
                app_name=_APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
        except Exception:
            session = None

        if not session:
            session = await svc.create_session(
                app_name=_APP_NAME,
                user_id=user_id,
            )
            session_id = session.id

        # Build user message
        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=message)],
        )

        full_text = ""
        images: list[dict] = []
        map_data: dict | None = None

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        full_text += part.text

        # Extract images and map data from the response
        images = _extract_images_from_text(full_text)
        map_data = _extract_map_data(full_text)

        # Try to extract place name for map if no coordinates found
        if not map_data:
            place = _extract_place_name(full_text)
            if place:
                map_data = {"place": place}

        return {
            "response": full_text,
            "images": images,
            "map_data": map_data,
            "session_id": session_id,
        }

    async def run_agent_stream(
        self,
        message: str,
        session_id: str,
        user_id: str = "default_user",
    ) -> AsyncGenerator[dict, None]:
        """
        Stream agent response chunks back.
        Yields dicts with type: text | image | map | status
        """
        runner = _get_runner()
        svc = _get_session_service()

        # Ensure session exists
        try:
            session = await svc.get_session(
                app_name=_APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
        except Exception:
            session = None

        if not session:
            session = await svc.create_session(
                app_name=_APP_NAME,
                user_id=user_id,
            )
            session_id = session.id

        # Build user message
        user_content = genai_types.Content(
            role="user",
            parts=[genai_types.Part.from_text(text=message)],
        )

        yield {"type": "status", "content": "Processing your request..."}

        full_text = ""

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content,
        ):
            # Only yield text from the final model response, not tool calls/responses
            if not event.content or not event.content.parts:
                continue

            # Determine which agent authored this event
            agent_author = getattr(event, "author", None) or "unknown"

            for part in event.content.parts:
                # ── Emit agent_event for debug panel, then skip ──
                # Function call (tool invocation)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    yield {
                        "type": "agent_event",
                        "data": {
                            "event_type": "tool_call",
                            "agent": agent_author,
                            "tool": getattr(fc, "name", str(fc)),
                            "args": (
                                dict(fc.args)
                                if hasattr(fc, "args") and fc.args
                                else {}
                            ),
                        },
                    }
                    continue

                # Function response (tool result)
                if hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    raw_resp = (
                        dict(fr.response)
                        if hasattr(fr, "response") and fr.response
                        else {}
                    )
                    # Truncate large responses for the debug panel
                    resp_str = str(raw_resp)
                    if len(resp_str) > 500:
                        resp_str = resp_str[:500] + "…"
                    yield {
                        "type": "agent_event",
                        "data": {
                            "event_type": "tool_response",
                            "agent": agent_author,
                            "tool": getattr(fr, "name", str(fr)),
                            "result": resp_str,
                        },
                    }
                    continue

                # Executable code
                if hasattr(part, "executable_code") and part.executable_code:
                    yield {
                        "type": "agent_event",
                        "data": {
                            "event_type": "code_exec",
                            "agent": agent_author,
                            "code": str(
                                part.executable_code.code
                                if hasattr(part.executable_code, "code")
                                else part.executable_code
                            )[:400],
                        },
                    }
                    continue

                # Code execution result
                if hasattr(part, "code_execution_result") and part.code_execution_result:
                    yield {
                        "type": "agent_event",
                        "data": {
                            "event_type": "code_result",
                            "agent": agent_author,
                            "output": str(
                                part.code_execution_result.output
                                if hasattr(part.code_execution_result, "output")
                                else part.code_execution_result
                            )[:400],
                        },
                    }
                    continue

                if hasattr(part, "text") and part.text:
                    chunk_text = part.text
                    # Filter out lines that look like tool call params/JSON args
                    chunk_text = _clean_tool_call_text(chunk_text)
                    if chunk_text.strip():
                        full_text += chunk_text
                        yield {"type": "text", "content": chunk_text}

        # After full response, extract structured data
        images = _extract_images_from_text(full_text)
        for img in images:
            yield {"type": "image", "data": img}

        map_data = _extract_map_data(full_text)
        if not map_data:
            place = _extract_place_name(full_text)
            if place:
                map_data = {"place": place}

        if map_data:
            yield {"type": "map", "data": map_data}
