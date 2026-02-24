# Register Ollama models with ADK via LiteLLM backend
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.registry import LLMRegistry
LLMRegistry._register("ollama/.*", LiteLlm)
LLMRegistry._register("ollama_chat/.*", LiteLlm)

from . import agent
