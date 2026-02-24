import logging
from typing import Any, Optional
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from database.db_manager import get_session

logger = logging.getLogger(__name__)

class PostgresSessionService(InMemorySessionService):
    """
    Extends InMemorySessionService to attempt loading missing sessions
    from the PostgreSQL database.
    """

    def get_session_state(self, session_id: str) -> Any:
        """
        Lookup session in memory first, then try database.
        """
        # 1. Check in-memory cache first
        if session_id in self._sessions:
            return self._sessions[session_id]

        # 2. Try to load from database
        try:
            history = get_session(session_id)
        except Exception as exc:
            logger.warning("DB session lookup failed for %s: %s", session_id, exc)
            history = None
        if history:
            logger.info(f"Session {session_id} found in DB. Loading context...")
            
            # Convert DB history (list of dicts) back to ADK Content objects
            adk_history = []
            for item in history:
                role = item.get("role")
                parts_raw = item.get("parts", [])
                parts = []
                for p in parts_raw:
                    if "text" in p:
                        parts.append(Part(text=p["text"]))
                    # Handle other part types if necessary
                
                adk_history.append(Content(role=role, parts=parts))

            # Store loaded history in in-memory cache so subsequent lookups are fast
            self._sessions[session_id] = adk_history
            return adk_history

        # 3. Not in memory or DB -> create new
        return super().get_session_state(session_id)
