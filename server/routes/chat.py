"""
routes/chat.py — Chat endpoint with SSE streaming.

POST /api/chat   → send a message, get SSE stream back
POST /api/chat/sync → send a message, get full response (non-streaming)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from connectors.agent_runner import AgentRunnerController

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

# Singleton controller
_controller = AgentRunnerController()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str = "default_user"


class ChatResponse(BaseModel):
    session_id: str
    response: str
    images: list[dict] = Field(default_factory=list)
    map_data: dict | None = None


@router.post("/chat")
async def chat_stream(req: ChatRequest):
    """Send a message and receive SSE-streamed response chunks."""
    session_id = req.session_id or str(uuid.uuid4())

    async def event_stream() -> AsyncGenerator[str, None]:
        full_response = ""
        images: list[dict] = []
        map_data: dict | None = None

        try:
            async for chunk in _controller.run_agent_stream(
                message=req.message,
                session_id=session_id,
                user_id=req.user_id,
            ):
                chunk_type = chunk.get("type", "text")

                if chunk_type == "text":
                    text = chunk.get("content", "")
                    full_response += text
                    yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

                elif chunk_type == "image":
                    images.append(chunk.get("data", {}))
                    yield f"data: {json.dumps({'type': 'image', 'data': chunk.get('data', {})})}\n\n"

                elif chunk_type == "map":
                    map_data = chunk.get("data", {})
                    yield f"data: {json.dumps({'type': 'map', 'data': map_data})}\n\n"

                elif chunk_type == "status":
                    yield f"data: {json.dumps({'type': 'status', 'content': chunk.get('content', '')})}\n\n"

            # Final done event with complete data
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'images': images, 'map_data': map_data})}\n\n"

        except Exception as e:
            logger.exception("Chat stream error")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
        },
    )


@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(req: ChatRequest):
    """Send a message and wait for the full response (non-streaming)."""
    session_id = req.session_id or str(uuid.uuid4())

    result = await _controller.run_agent(
        message=req.message,
        session_id=session_id,
        user_id=req.user_id,
    )

    return ChatResponse(
        session_id=session_id,
        response=result.get("response", ""),
        images=result.get("images", []),
        map_data=result.get("map_data"),
    )
