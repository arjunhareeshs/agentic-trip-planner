"""
server/main.py — Custom FastAPI server wrapping the ADK agent runner.

Exposes REST + SSE endpoints for the React frontend to interact with the
trip planner agent without using ADK's built-in web UI.

Run with:
    cd "c:\\agentic ai\\agent-trip-planner\\server"
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Ensure the src package and server root are importable
server_root = Path(__file__).parent
sys.path.insert(0, str(server_root / "src"))
sys.path.insert(0, str(server_root))

# Load .env before anything else
from dotenv import load_dotenv
_ENV = Path(__file__).parent / "src" / "agents" / ".env"
load_dotenv(_ENV)

# Register Ollama models with ADK (must happen before agent import)
from google.adk.models.lite_llm import LiteLlm  # type: ignore[import-untyped]
from google.adk.models.registry import LLMRegistry  # type: ignore[import-untyped]
LLMRegistry._register("ollama/.*", LiteLlm)
LLMRegistry._register("ollama_chat/.*", LiteLlm)

from middlewares.cors import add_cors_middleware
from routes.chat import router as chat_router
from routes.session import router as session_router
from routes.images import router as images_router
from routes.geocode import router as geocode_router
from routes.upload import router as upload_router
from routes.voice import router as voice_router

logger = logging.getLogger(__name__)

# ── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Trip Planner API starting up …")
    yield
    logger.info("Trip Planner API shutting down …")

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Trip Planner API",
    version="1.0.0",
    description="REST + SSE backend for the Trip Planner frontend",
    lifespan=lifespan,
)

# CORS
add_cors_middleware(app)

# Routes
app.include_router(chat_router,    prefix="/api")
app.include_router(session_router, prefix="/api")
app.include_router(images_router,  prefix="/api")
app.include_router(geocode_router, prefix="/api")
app.include_router(upload_router,  prefix="/api")
app.include_router(voice_router,   prefix="/api")



@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "trip-planner-api"}
