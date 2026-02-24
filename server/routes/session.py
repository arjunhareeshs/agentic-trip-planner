"""
routes/session.py — Session management endpoints.

POST /api/session        → create a new session
GET  /api/session/{id}   → retrieve session info
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from pydantic import BaseModel

from connectors.agent_runner import AgentRunnerController

router = APIRouter(tags=["session"])
_controller = AgentRunnerController()


class SessionCreateResponse(BaseModel):
    session_id: str
    user_id: str


@router.post("/session", response_model=SessionCreateResponse)
async def create_session(user_id: str = "default_user"):
    """Create a fresh agent session."""
    session_id = str(uuid.uuid4())
    await _controller.create_session(session_id=session_id, user_id=user_id)
    return SessionCreateResponse(session_id=session_id, user_id=user_id)


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Retrieve session metadata (turns, state keys)."""
    info = await _controller.get_session_info(session_id)
    return info
