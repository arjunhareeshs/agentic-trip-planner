"""
agents/memory/services.py

Singleton instances of the ADK session and memory services.

Both are in-memory implementations that persist for the lifetime of the
running process.  Swap these out for VertexAiSessionService /
VertexAiMemoryBankService to persist across restarts.
"""

import logging

from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService

logger = logging.getLogger(__name__)

# ── Session service ─────────────────────────────────────────────────────────
# Stores the full conversation state (turn history + state dict) per session.
# Attempts to use the PostgresSessionService; falls back to the plain
# InMemorySessionService if Postgres or ADK internals are unavailable.
try:
    from .postgres_session_service import PostgresSessionService
    session_service = PostgresSessionService()
    logger.debug("Using PostgresSessionService for session persistence.")
except Exception as _exc:
    logger.warning(
        "PostgresSessionService unavailable (%s). Falling back to InMemorySessionService.",
        _exc,
    )
    session_service = InMemorySessionService()


# ── Memory service ──────────────────────────────────────────────────────────
# Stores compressed long-term memories extracted from past sessions.
# add_session_to_memory()  writes the current session into the memory bank.
# search_memory(query)     retrieves the most relevant snippets.
memory_service = InMemoryMemoryService()
