"""
╔══════════════════════════════════════════════════════════════╗
║          Dual-Modal RAG Pipeline — server/src/RAG            ║
║                                                              ║
║  Public API (the ONLY symbols external code may import):     ║
║    • RAGPipeline  — ingest() and query() entry points        ║
║    • RAGResult    — typed return wrapper                      ║
║    • RAGContext   — assembled retrieval context               ║
╚══════════════════════════════════════════════════════════════╝

External modules must ONLY import from this file.
Never import internal submodules directly from outside this package.
"""

__version__ = "1.0.0"
__author__ = "Agent Trip Planner"

from .pipeline import RAGPipeline
from .rag_types import RAGResult, RAGContext

__all__ = ["RAGPipeline", "RAGResult", "RAGContext"]
