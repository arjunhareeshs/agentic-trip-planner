"""
agents/memory — Context window and session memory management.

Exports the four ADK callback functions and the configured memory/session
services that are wired into the agent definitions.
"""

from .context_manager import (
    before_model_callback,
    after_root_agent_callback,
    after_subagent_callback,
    before_tool_callback,
    after_tool_callback,
)
from .services import memory_service, session_service

__all__ = [
    "before_model_callback",
    "after_root_agent_callback",
    "after_subagent_callback",
    "before_tool_callback",
    "after_tool_callback",
    "memory_service",
    "session_service",
]
