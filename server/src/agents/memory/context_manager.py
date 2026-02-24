"""
agents/memory/context_manager.py

Context-window management and session memory for TripCraft agents.

WHAT THIS MODULE DOES
═════════════════════

1. TOOL RESULT COMPRESSION  (after_tool_callback)
   Every tool response is inspected before it enters the message history.
   Responses larger than MAX_TOOL_RESULT_CHARS are truncated to the most
   useful portion (first N destinations / lines).  The full result is
   cached in session state so later turns can re-read it without re-calling
   the tool.

2. KG TOOL CACHING  (before_tool_callback)
   Knowledge-graph tool calls are expensive and deterministic.  If an
   identical call (same tool + same arguments) has already been made in
   this session, the cached result is returned immediately — no second
   network/compute round-trip.

3. CONTEXT WINDOW MANAGEMENT  (before_model_callback)
   Before every LLM call on the root orchestrator:
   a. Oversized raw tool results still in `llm_request.contents` are
      trimmed again (belt-and-suspenders).
   b. If the total character count of all messages exceeds
      MAX_CONTEXT_CHARS, the oldest middle turns are dropped (the system
      prompt and last KEEP_RECENT_TURNS turns are always preserved).
   c. If a rolling conversation summary exists in state, it is prepended
      as a synthetic context message so the model always has the key facts.
   d. Structured sub-agent outputs (preference results, itinerary, booking)
      stored in state are summarised and injected as a compact context block.

4. EXCHANGE COUNTING + MEMORY PERSISTENCE  (after_root_agent_callback)
   After each complete root-agent response:
   • The exchange counter in state is incremented.
   • Every SUMMARY_EVERY_N_EXCHANGES (default 30) exchanges the session is
     saved to the long-term memory service and a compact rolling summary of
     the conversation (destination, preferences, itinerary highlights,
     booking status) is written to `state[KEY_SUMMARY]`.

5. STRUCTURED STATE EXTRACTION  (after_subagent_callback)
   After each sub-agent completes, key facts are extracted from its output
   and written to named state keys so all subsequent agents can reference
   them without re-parsing long output strings.

STATE KEY REFERENCE
═══════════════════
  exchange_count        int     Running count of complete user exchanges.
  conv_summary          str     Compact rolling summary, updated every 30 turns.
  pref_profile          dict    Extracted user preference profile.
  confirmed_dest        str     The destination the user chose.
  itinerary_highlights  str     One-paragraph itinerary summary.
  booking_status        str     Flight/hotel booking summary line.
  tool_cache            dict    {cache_key: result} for KG tool results.
  tool_full_results     dict    {cache_key: full_result} pre-compression.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Optional

import google.genai.types as genai_types  # type: ignore[import-untyped]
from google.adk.agents.callback_context import CallbackContext  # type: ignore[import-untyped]
from google.adk.models import LlmRequest, LlmResponse  # type: ignore[import-untyped]
from google.adk.tools.base_tool import BaseTool  # type: ignore[import-untyped]
from google.adk.tools.tool_context import ToolContext  # type: ignore[import-untyped]

# Database integration (absolute import — database/ is a sibling of agents/ on sys.path)
from database.db_manager import save_or_update_session, save_or_update_plan, db_available as _db_available

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

SUMMARY_EVERY_N_EXCHANGES: int = 30   # trigger memory save + summary

MAX_TOOL_RESULT_CHARS: int = 4_000    # tool response trimmed beyond this
TRIM_TOOL_RESULT_TO: int = 1_500      # how much to keep after trimming

MAX_CONTEXT_CHARS: int = 120_000      # total context char budget
KEEP_RECENT_TURNS: int = 20           # minimum recent turns to preserve

# KG tools whose results are safe to cache (deterministic, no side effects)
CACHEABLE_TOOLS = frozenset({
    "match_destinations",
    "filter_destinations",
    "get_destination_details",
    "list_all_destinations",
    "get_graph_stats",
})

# ── State keys ───────────────────────────────────────────────────────────────

KEY_EXCHANGE_COUNT    = "exchange_count"
KEY_SUMMARY           = "conv_summary"
KEY_PREF_PROFILE      = "pref_profile"
KEY_CONFIRMED_DEST    = "confirmed_dest"
KEY_ITINERARY_HIGHLIGHTS = "itinerary_highlights"
KEY_BOOKING_STATUS    = "booking_status"
KEY_WEATHER_ADVISORY  = "weather_advisory"
KEY_TOOL_CACHE        = "tool_cache"
KEY_TOOL_FULL         = "tool_full_results"

# ADK sub-agent output_key values (written into state automatically by ADK)
ADK_PREFERENCE_KEY    = "preference_results"
ADK_ITINERARY_KEY     = "itinerary_result"
ADK_BOOKING_KEY       = "booking_result"
ADK_IMAGE_KEY         = "image_analysis_result"


# ═══════════════════════════════════════════════════════════════════════════
# 1. BEFORE TOOL  — cache lookup for KG tools
# ═══════════════════════════════════════════════════════════════════════════

def before_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Optional[dict]:
    """
    Return a cached result instead of re-calling the tool when the same KG
    tool is invoked with identical arguments in the same session.
    """
    if tool.name not in CACHEABLE_TOOLS:
        return None  # not cacheable — let tool execute normally

    cache_key = _make_cache_key(tool.name, args)
    cache: dict = tool_context.state.get(KEY_TOOL_CACHE) or {}

    if cache_key in cache:
        logger.debug("Cache HIT for %s (%s)", tool.name, cache_key[:12])
        return cache[cache_key]  # return cached result, skip tool call

    return None  # cache miss — execute tool normally


# ═══════════════════════════════════════════════════════════════════════════
# 2. AFTER TOOL  — compress + cache every tool result
# ═══════════════════════════════════════════════════════════════════════════

def after_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    """
    1. Store the full result in state[KEY_TOOL_FULL] for potential recovery.
    2. If the result is larger than MAX_TOOL_RESULT_CHARS, produce a
       compressed version for the conversation history.
    3. Cache the (possibly compressed) result in state[KEY_TOOL_CACHE] for
       future before_tool_callback hits.
    """
    serialised = _serialise(tool_response)

    # ── Persist full result ──────────────────────────────────────────────
    if tool.name in CACHEABLE_TOOLS:
        cache_key = _make_cache_key(tool.name, args)
        full_store: dict = tool_context.state.get(KEY_TOOL_FULL) or {}

        # Compress before caching to avoid bloating state
        compressed_result = _compress_result(tool_response, len(serialised))

        # Write to cache (use compressed copy)
        cache: dict = tool_context.state.get(KEY_TOOL_CACHE) or {}
        cache[cache_key] = compressed_result
        tool_context.state[KEY_TOOL_CACHE] = cache

        # Write full result if it hasn't been stored before
        full_store[cache_key] = tool_response
        tool_context.state[KEY_TOOL_FULL] = full_store

        if len(serialised) > MAX_TOOL_RESULT_CHARS:
            logger.debug(
                "Compressed %s result: %d → %d chars",
                tool.name,
                len(serialised),
                len(_serialise(compressed_result)),
            )
            return compressed_result

    elif len(serialised) > MAX_TOOL_RESULT_CHARS:
        # Non-KG tool but still oversized — truncate text fields
        compressed_result = _compress_result(tool_response, len(serialised))
        logger.debug(
            "Compressed non-KG tool %s: %d → %d chars",
            tool.name,
            len(serialised),
            len(_serialise(compressed_result)),
        )
        return compressed_result

    return None  # result is fine as-is


# ═══════════════════════════════════════════════════════════════════════════
# 3. BEFORE MODEL  — context window management
# ═══════════════════════════════════════════════════════════════════════════

def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """
    Called before every LLM call on the root orchestrator.
    Manages the context window:
      a. Belt-and-suspenders trim of oversized tool results still in history.
      b. History truncation when total chars exceed MAX_CONTEXT_CHARS.
      c. Inject rolling conversation summary as synthetic leading context.
      d. Inject compact sub-agent result summaries.
    """
    contents = llm_request.contents
    if not contents:
        return None

    # a. Strip raw tool-call JSON text from model messages in history
    #    (prevents the model from seeing / repeating its own raw JSON)
    _clean_model_text_parts(contents)

    # b. Compress any oversized function-response parts still in history
    _compress_history_tool_results(contents)

    # c. Trim oldest middle turns if context is too long
    _trim_history_if_needed(contents)

    # d. Inject conversation summary (if any)
    _inject_conv_summary(callback_context, llm_request)

    # e. Inject sub-agent outputs as compact context
    _inject_subagent_context(callback_context, llm_request)

    return None  # always continue — never short-circuit the model call


# ═══════════════════════════════════════════════════════════════════════════
# 4. AFTER ROOT AGENT  — exchange counting + memory persistence
# ═══════════════════════════════════════════════════════════════════════════

def after_root_agent_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """
    Increment exchange counter.  Every SUMMARY_EVERY_N_EXCHANGES turns:
      1. Save the full session to the long-term memory bank.
      2. Build and store a compact rolling conversation summary.
    """
    state = callback_context.state

    count: int = state.get(KEY_EXCHANGE_COUNT, 0) + 1
    state[KEY_EXCHANGE_COUNT] = count

    logger.debug("Exchange #%d completed", count)

    if count % SUMMARY_EVERY_N_EXCHANGES == 0:
        logger.info(
            "Exchange threshold reached (%d). Saving session to memory and summarising.",
            count,
        )
        # Persist session events to long-term memory
        try:
            callback_context.add_session_to_memory()
        except Exception as exc:
            logger.warning("add_session_to_memory failed: %s", exc)

        # Build and store the rolling summary
        _update_rolling_summary(callback_context)

    # ── Persistent Storage — Save full conversation to PostgreSQL ───────
    if _db_available:
        try:
            session_id = callback_context.session.id
            # Convert session events to serializable history list
            history = []
            for event in callback_context.session.events:
                content = event.content
                if content is None or not content.parts:
                    continue
                parts = []
                for part in content.parts:
                    if part.text:
                        parts.append({"text": part.text})
                    elif part.function_call:
                        parts.append({"function_call": {
                            "name": part.function_call.name,
                            "args": part.function_call.args
                        }})
                    elif part.function_response:
                        parts.append({"function_response": {
                            "name": part.function_response.name,
                            "response": part.function_response.response
                        }})
                if parts:
                    role = content.role or event.author or "user"
                    history.append({"role": role, "parts": parts})
            
            save_or_update_session(session_id, history)
        except Exception as exc:
            logger.warning("Failed to save session to database: %s", exc)

    # ── Strip raw tool-call JSON from the agent's final response ────────
    # If the model emitted tool-call metadata as text, clean it so the
    # user only sees natural language.
    try:
        events = callback_context.session.events
        if events:
            # Walk backwards to find the last model message with text
            for evt in reversed(events):
                c = evt.content
                if c is None or getattr(c, 'role', None) != 'model' or not c.parts:
                    continue
                text_parts = [p for p in c.parts if p.text]
                if not text_parts:
                    continue
                if any(_has_tool_call_text(p.text) for p in text_parts):
                    cleaned_parts = []
                    for p in c.parts:
                        if p.text:
                            cleaned = _strip_tool_call_text(p.text)
                            if cleaned:
                                cleaned_parts.append(genai_types.Part(text=cleaned))
                        else:
                            cleaned_parts.append(p)
                    if cleaned_parts:
                        return genai_types.Content(role="model", parts=cleaned_parts)
                break  # only check the last relevant model event
    except Exception as exc:
        logger.debug("Could not clean tool-call text from response: %s", exc)

    return None  # do not modify the agent's response



# ═══════════════════════════════════════════════════════════════════════════
# 5. AFTER SUB-AGENT  — extract structured facts into state
# ═══════════════════════════════════════════════════════════════════════════

def after_subagent_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """
    After a sub-agent completes, extract key structured facts from the
    ADK-written output_key in session state and write them to named state
    keys so the orchestrator and subsequent agents can reference them cheaply.
    """
    state = callback_context.state
    # Use getattr with fallback — agent_name may not exist in all ADK versions
    agent = getattr(callback_context, "agent_name", None) or ""

    # ADK writes sub-agent output to state[output_key] automatically.
    # Map each sub-agent's output_key to its text for extraction.
    output_key_map = {
        "preference_agent": "preference_results",
        "itinerary_agent":  "itinerary_result",
        "booking_agent":    "booking_result",
        "image_analysis_agent": "image_analysis_result",
    }
    out_key = output_key_map.get(agent)
    raw = state.get(out_key, "") if out_key else ""
    response_text = raw if isinstance(raw, str) else _serialise(raw)
    if not response_text:
        return None

    if agent == "preference_agent":
        _extract_preference_facts(state, response_text)

    elif agent == "itinerary_agent":
        _extract_itinerary_facts(state, response_text)

    elif agent == "booking_agent":
        _extract_booking_facts(state, response_text)
        # ── Weather Safety Check ── automatically check climate hazards ────
        _run_weather_safety_check(state)

    # ── Persistent Storage — Save decision/plan to PostgreSQL ──────────
    if _db_available:
        try:
            session_id = callback_context.session.id
            dest = state.get(KEY_CONFIRMED_DEST)
            if dest:
                # Gather available details
                plan_details = {
                    "itinerary": state.get(KEY_ITINERARY_HIGHLIGHTS),
                    "booking": state.get(KEY_BOOKING_STATUS),
                    "preference_profile": state.get(KEY_PREF_PROFILE)
                }
                # Extract images from history if any (from search_place_images results)
                images = []
                # Optionally check tool_full_results for search_place_images hits
                full_results = state.get(KEY_TOOL_FULL, {})
                for key, res in full_results.items():
                    if isinstance(res, list) and len(res) > 0 and "image_url" in res[0]:
                        images.extend(res)
                
                # Save or update the plan in DB
                save_or_update_plan(
                    session_id=session_id,
                    destination=dest,
                    plan_details=plan_details,
                    images=images,
                    confirmed=(agent == "booking_agent" or state.get(KEY_BOOKING_STATUS) is not None)
                )
        except Exception as exc:
            logger.warning("Failed to save plan to database: %s", exc)

    return None  # do not modify the sub-agent's response



# ═══════════════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _make_cache_key(tool_name: str, args: dict) -> str:
    """Stable hash key from tool name + sorted arg values."""
    payload = json.dumps({"t": tool_name, "a": args}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def _serialise(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


# ── Raw tool-call text stripping ─────────────────────────────────────────────
#
# DeepSeek (and some other models via Ollama/LiteLLM) sometimes emit
# tool-call metadata as plain text alongside proper function_call parts.
# This raw JSON pollutes the chat and confuses the model on later turns.
# The helpers below detect and strip it.

_RE_TOOL_CALLS_BLOCK = re.compile(
    r'Tool\s+Calls?\s*:\s*\[',
    re.IGNORECASE,
)

_RE_TOOL_CALL_XML = re.compile(
    r'</?tool_call>.*?(?:</tool_call>|$)',
    re.DOTALL | re.IGNORECASE,
)


def _has_tool_call_text(text: str) -> bool:
    """Return True if *text* contains raw tool-call JSON / metadata."""
    if not text:
        return False
    return bool(
        _RE_TOOL_CALLS_BLOCK.search(text)
        or _RE_TOOL_CALL_XML.search(text)
        or re.search(r'"type"\s*:\s*"function"\s*,\s*"function"\s*:', text)
    )


def _strip_tool_call_text(text: str) -> str:
    """
    Remove raw tool-call JSON / metadata that models sometimes emit as text.
    Returns only the natural-language portions.
    """
    if not text:
        return text

    # 1. "Tool Calls: [...]" — bracket-match the JSON array
    match = _RE_TOOL_CALLS_BLOCK.search(text)
    if match:
        start = match.start()
        depth, i = 0, match.end() - 1
        while i < len(text):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    text = text[:start].rstrip() + '\n' + text[i + 1:].lstrip()
                    break
            i += 1
        else:
            text = text[:start].rstrip()

    # 2. <tool_call>...</tool_call> XML blocks
    text = _RE_TOOL_CALL_XML.sub('', text)

    # 3. Collapse excess blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _clean_model_text_parts(contents: list[genai_types.Content]) -> None:
    """
    Walk through every model message in *contents* and strip raw
    tool-call JSON from text parts.  Mutates in-place.
    This prevents the model from seeing (and repeating) its own raw
    tool-call outputs on subsequent turns, breaking the repetition cycle.
    """
    for content in contents:
        if not content.parts:
            continue
        # Only clean model messages (role="model")
        if getattr(content, 'role', None) != 'model':
            continue
        for part in content.parts:
            if part.text and _has_tool_call_text(part.text):
                cleaned = _strip_tool_call_text(part.text)
                try:
                    part.text = cleaned if cleaned else ""
                except (AttributeError, TypeError):
                    pass  # proto-backed immutable Part


# ── Tool result compression ──────────────────────────────────────────────────

def _compress_result(result: Any, original_len: int) -> Any:
    """
    Produce a compressed representation of a tool result.

    For list results (e.g. match_destinations returns a list of dicts):
      - Keep only the first 5 items.
      - Truncate each item's long string fields to 200 chars.

    For dict results (e.g. get_destination_details):
      - Keep all keys but truncate any string value > 300 chars.

    For string results:
      - Truncate to TRIM_TOOL_RESULT_TO chars.
    """
    if isinstance(result, list):
        truncated = result[:5]
        return [_trim_dict_values(item, 200) if isinstance(item, dict) else item
                for item in truncated]

    if isinstance(result, dict):
        # Special case: if it's an error dict, keep it intact
        if "error" in result:
            return result
        return _trim_dict_values(result, 300)

    if isinstance(result, str) and len(result) > TRIM_TOOL_RESULT_TO:
        return result[:TRIM_TOOL_RESULT_TO] + f"... [truncated {original_len} chars]"

    return result


def _trim_dict_values(d: dict, max_len: int) -> dict:
    """Truncate string values in a dict that exceed max_len characters."""
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > max_len:
            out[k] = v[:max_len] + "..."
        elif isinstance(v, list) and len(v) > 10:
            out[k] = v[:10]
        else:
            out[k] = v
    return out


# ── History compression ──────────────────────────────────────────────────────

def _compress_history_tool_results(
    contents: list[genai_types.Content],
) -> None:
    """
    Iterate through every Part in history and truncate function_response
    payloads that exceed MAX_TOOL_RESULT_CHARS.  Mutates in-place.
    """
    for content in contents:
        if not content.parts:
            continue
        for part in content.parts:
            if part.function_response is None:
                continue
            resp = part.function_response.response
            if resp is None:
                continue
            serialised = _serialise(resp)
            if len(serialised) > MAX_TOOL_RESULT_CHARS:
                compressed = _compress_result(resp, len(serialised))
                try:
                    part.function_response.response = compressed
                except (AttributeError, TypeError):
                    # Proto-backed Part objects may not support in-place mutation;
                    # log and skip so the original oversized response is kept.
                    logger.debug(
                        "Could not mutate proto-backed Part.function_response; skipping compression."
                    )


def _trim_history_if_needed(
    contents: list[genai_types.Content],
) -> None:
    """
    If the total character count of all contents exceeds MAX_CONTEXT_CHARS,
    remove the oldest middle turns while preserving:
      - Everything before index KEEP_SYSTEM_MESSAGES (the system/instruction context).
      - The last KEEP_RECENT_TURNS messages.

    Removal is done in pairs (user + model) to keep conversation coherent.
    """
    KEEP_SYSTEM_MESSAGES = 1   # preserve the leading system content

    def _part_char_count(p: genai_types.Part) -> int:
        """Estimate character cost of any part type."""
        if p.text:
            return len(p.text)
        if p.function_call:
            try:
                return len(json.dumps({"n": p.function_call.name, "a": p.function_call.args}, default=str))
            except Exception:
                return 128
        if p.function_response:
            try:
                return len(json.dumps({"n": p.function_response.name, "r": p.function_response.response}, default=str))
            except Exception:
                return 256
        return 0

    total_chars = sum(
        sum(_part_char_count(p) for p in c.parts)
        for c in contents
        if c.parts
    )

    if total_chars <= MAX_CONTEXT_CHARS:
        return

    logger.info(
        "Context too large (%d chars). Trimming history.", total_chars
    )

    # Identify removable range: skip first KEEP_SYSTEM_MESSAGES and last KEEP_RECENT_TURNS
    removable_start = KEEP_SYSTEM_MESSAGES
    removable_end   = max(removable_start, len(contents) - KEEP_RECENT_TURNS)

    if removable_end <= removable_start:
        return  # nothing to remove

    # Remove oldest turns in the removable window (remove in pairs)
    num_to_remove = min(10, removable_end - removable_start)
    del contents[removable_start: removable_start + num_to_remove]

    logger.debug("Removed %d turns from history.", num_to_remove)


# ── Context injection ────────────────────────────────────────────────────────

def _inject_conv_summary(
    ctx: CallbackContext,
    llm_request: LlmRequest,
) -> None:
    """
    Prepend the rolling conversation summary as a synthetic context message
    immediately after the system instruction (index 1).
    Skipped if no summary exists yet.
    """
    summary: str = ctx.state.get(KEY_SUMMARY, "")
    if not summary:
        return

    context_text = (
        "[PRIOR CONVERSATION CONTEXT — key facts so far]\n"
        f"{summary}\n"
        "[END PRIOR CONTEXT]"
    )

    synthetic = genai_types.Content(
        role="model",
        parts=[genai_types.Part(text=context_text)],
    )

    insert_pos = min(1, len(llm_request.contents))
    llm_request.contents.insert(insert_pos, synthetic)


def _inject_subagent_context(
    ctx: CallbackContext,
    llm_request: LlmRequest,
) -> None:
    """
    Build a compact block from structured sub-agent state keys and inject it
    just before the latest user message so the model always has critical
    planning facts in the near-context window.
    Skipped entirely if no sub-agent results are stored yet.
    """
    lines: list[str] = []

    pref = ctx.state.get(KEY_PREF_PROFILE)
    if pref:
        lines.append(f"User profile: {json.dumps(pref, ensure_ascii=False)}")

    dest = ctx.state.get(KEY_CONFIRMED_DEST)
    if dest:
        lines.append(f"Confirmed destination: {dest}")

    itin = ctx.state.get(KEY_ITINERARY_HIGHLIGHTS)
    if itin:
        lines.append(f"Itinerary summary: {itin}")

    book = ctx.state.get(KEY_BOOKING_STATUS)
    if book:
        lines.append(f"Booking status: {book}")

    weather = ctx.state.get(KEY_WEATHER_ADVISORY)
    if weather:
        lines.append(f"Weather advisory: {weather}")

    if not lines:
        return

    context_text = "[SESSION STATE]\n" + "\n".join(lines) + "\n[END SESSION STATE]"

    synthetic = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=context_text)],
    )

    # Insert just before the last user message
    if llm_request.contents:
        llm_request.contents.insert(
            max(0, len(llm_request.contents) - 1),
            synthetic,
        )


# ── Rolling summary ──────────────────────────────────────────────────────────

def _update_rolling_summary(ctx: CallbackContext) -> None:
    """
    Build a compact factual summary of the current session and store it in
    state[KEY_SUMMARY].  Relies entirely on structured state keys — no LLM
    call needed.
    """
    state = ctx.state
    parts: list[str] = []

    count  = state.get(KEY_EXCHANGE_COUNT, 0)
    parts.append(f"Session has had {count} exchanges.")

    pref = state.get(KEY_PREF_PROFILE)
    if pref:
        parts.append(f"Traveller profile: {json.dumps(pref, ensure_ascii=False)}")

    dest = state.get(KEY_CONFIRMED_DEST)
    if dest:
        parts.append(f"Chosen destination: {dest}.")

    # Preference agent output (ADK writes it under output_key)
    pref_out = state.get(ADK_PREFERENCE_KEY)
    if pref_out and isinstance(pref_out, str):
        # Keep only the first 400 chars of the preference result
        snippet = pref_out[:400] + ("..." if len(pref_out) > 400 else "")
        parts.append(f"Destination options presented: {snippet}")

    itin = state.get(KEY_ITINERARY_HIGHLIGHTS)
    if itin:
        parts.append(f"Itinerary highlights: {itin}")

    book = state.get(KEY_BOOKING_STATUS)
    if book:
        parts.append(f"Booking: {book}")

    summary = " | ".join(parts)
    state[KEY_SUMMARY] = summary
    logger.info("Rolling summary updated: %s", summary[:120])


# ── Sub-agent fact extraction ────────────────────────────────────────────────

def _extract_text(content: genai_types.Content) -> str:
    """Pull all text parts from a Content object."""
    if content is None:
        return ""
    parts = content.parts or []
    return " ".join(p.text or "" for p in parts if p.text).strip()


def _extract_preference_facts(state: dict, text: str) -> None:
    """
    Parse the preference agent's output for the top-ranked destination
    and update the user preference profile in state.
    """
    # Extract destination names (lines starting with rank number)
    destinations: list[str] = re.findall(
        r"^\d+\.\s+\*\*([^\*]+)\*\*", text, re.MULTILINE
    )
    if destinations:
        # If only one option was produced, auto-set as preference
        if len(destinations) == 1:
            state[KEY_CONFIRMED_DEST] = destinations[0].strip()

    # Extract emotions/vibes mentioned
    vibes = re.findall(r"Vibe:\s*([^\n]+)", text)
    profile: dict = state.get(KEY_PREF_PROFILE) or {}
    if vibes:
        profile["vibes"] = [v.strip() for v in vibes[:3]]
    if destinations:
        profile["top_destinations"] = [d.strip() for d in destinations[:4]]
    state[KEY_PREF_PROFILE] = profile


def _extract_itinerary_facts(state: dict, text: str) -> None:
    """
    Extract destination and cost summary from itinerary agent output.
    """
    # Destination from the header line "# [Destination] — N-Day Itinerary"
    dest_match = re.search(r"#\s+(.+?)\s+[—-]\s+\d+", text)
    if dest_match:
        state[KEY_CONFIRMED_DEST] = dest_match.group(1).strip()

    # Total cost from the summary table
    total_match = re.search(
        r"\|\s*\*{0,2}TOTAL\*{0,2}\s*\|\s*\*{0,2}([^\|]+)\*{0,2}\s*\|",
        text, re.IGNORECASE,
    )
    if total_match:
        state[KEY_ITINERARY_HIGHLIGHTS] = (
            f"Total trip cost: {total_match.group(1).strip()}"
        )
    else:
        # Fallback: take first 300 chars of the itinerary
        state[KEY_ITINERARY_HIGHLIGHTS] = text[:300].replace("\n", " ").strip()


def _extract_booking_facts(state: dict, text: str) -> None:
    """
    Extract a one-line booking status summary from booking agent output.
    """
    # Look for "Flights: X to Y" header
    flight_match = re.search(r"##\s+Flights?:\s+(.+)", text, re.IGNORECASE)
    hotel_count  = len(re.findall(r"^\d+\.\s+\*\*", text, re.MULTILINE))

    parts: list[str] = []
    if flight_match:
        parts.append(f"Flights searched: {flight_match.group(1).strip()}")
    if hotel_count:
        parts.append(f"{hotel_count} hotel option(s) found")

    if parts:
        state[KEY_BOOKING_STATUS] = " | ".join(parts)
    elif text:
        state[KEY_BOOKING_STATUS] = text[:200].replace("\n", " ").strip()


# ── Weather Safety Check ────────────────────────────────────────────────────────────────

# Hazardous weather keywords that signal danger
_HAZARD_KEYWORDS = frozenset({
    "thunderstorm", "heavy rain", "heavy snow", "tornado", "hurricane",
    "cyclone", "typhoon", "blizzard", "hail", "flood", "extreme",
    "storm", "severe", "squall", "tropical storm", "dust storm",
    "sandstorm", "ice storm", "freezing rain",
})

# Temperature thresholds (Celsius)
_EXTREME_HEAT_C = 45.0
_EXTREME_COLD_C = -15.0
_HIGH_WIND_MPS = 20.0  # ~72 km/h
_HEAVY_RAIN_MM = 30.0  # per 3-hour period


def _run_weather_safety_check(state: dict) -> None:
    """
    After a booking is confirmed, automatically check the weather forecast
    for the destination. If hazardous conditions are detected, store a
    warning in state[KEY_WEATHER_ADVISORY] so the orchestrator can alert
    the user and suggest withdrawing/postponing the booking.
    """
    dest = state.get(KEY_CONFIRMED_DEST)
    if not dest:
        return

    try:
        from agents.tools.api_connectors.geoapify import geocode
        from agents.tools.api_connectors.openweather import get_weather_forecast

        # Step 1: Geocode the destination
        geo = geocode(dest)
        if "error" in geo or not geo.get("lat"):
            logger.debug("Weather safety check: could not geocode '%s'", dest)
            return

        lat, lng = geo["lat"], geo["lng"]

        # Step 2: Get 5-day forecast
        forecasts = get_weather_forecast(lat, lng, days=5)
        if not forecasts or (len(forecasts) == 1 and "error" in forecasts[0]):
            logger.debug("Weather safety check: no forecast data for '%s'", dest)
            return

        # Step 3: Scan for hazards
        warnings: list[str] = []
        for day in forecasts:
            date = day.get("date", "unknown")
            weather = (day.get("weather", "") or "").lower()
            temp = day.get("temp_c", 25)
            wind = day.get("wind_speed_mps", 0)
            rain = day.get("rain_mm", 0)

            # Check weather description for hazard keywords
            for kw in _HAZARD_KEYWORDS:
                if kw in weather:
                    warnings.append(f"{date}: {weather.title()} expected")
                    break

            # Extreme temperature
            if temp >= _EXTREME_HEAT_C:
                warnings.append(f"{date}: Extreme heat {temp}°C")
            elif temp <= _EXTREME_COLD_C:
                warnings.append(f"{date}: Extreme cold {temp}°C")

            # High wind
            if wind >= _HIGH_WIND_MPS:
                warnings.append(f"{date}: Dangerous winds {wind} m/s ({wind * 3.6:.0f} km/h)")

            # Heavy rainfall
            if rain >= _HEAVY_RAIN_MM:
                warnings.append(f"{date}: Heavy rainfall {rain}mm")

        # Step 4: Store advisory in state
        if warnings:
            advisory = (
                f"⚠️ WEATHER HAZARD ALERT for {dest}: "
                + "; ".join(warnings[:5])
                + ". Consider postponing or withdrawing your booking "
                  "until conditions improve."
            )
            state[KEY_WEATHER_ADVISORY] = advisory
            logger.warning("Weather safety alert for %s: %s", dest, advisory)
        else:
            state[KEY_WEATHER_ADVISORY] = (
                f"✅ Weather looks safe for {dest} over the next 5 days. "
                "No hazardous conditions detected."
            )
            logger.info("Weather safety check passed for %s", dest)

    except Exception as exc:
        logger.warning("Weather safety check failed: %s", exc)
