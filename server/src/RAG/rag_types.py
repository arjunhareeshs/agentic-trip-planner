"""
types.py — Shared, frozen dataclasses for ALL inter-module data contracts.

RULES:
  • Every inter-module data transfer uses one of these types.
  • All dataclasses are frozen=True (immutable) to prevent silent mutation.
  • No raw dicts or loose strings cross module boundaries.
  • Importing this file has ZERO side effects (no numpy, no torch at import time).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════
#  PARSING TYPES
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DocumentElement:
    """
    A single structural element extracted from a PDF by the parser.
    element_type: heading | paragraph | table | list | image_caption |
                  pricing_block | map | unknown
    level: heading depth (1 = H1, 2 = H2, …), 0 for non-headings
    """
    element_type: str
    content: str
    page_number: int
    element_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: int = 0
    bbox: Optional[tuple] = None           # (x0, y0, x1, y1) bounding box
    table_data: Optional[List[List[str]]] = None  # structured rows for tables
    extra_metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ParsedDocument:
    """Complete output of the PDF parser stage."""
    source_pdf: str                        # absolute path to the PDF
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    elements: tuple = field(default_factory=tuple)   # tuple of DocumentElement
    total_pages: int = 0
    parse_warnings: tuple = field(default_factory=tuple)   # non-fatal issues


@dataclass(frozen=True)
class ExtractedImage:
    """A single image extracted from a PDF page by fitz."""
    image_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image_bytes: bytes = field(default=b"")
    page_number: int = 0
    source_pdf: str = ""
    caption: str = ""
    bbox: Optional[tuple] = None          # image bounding box on page
    width: int = 0
    height: int = 0
    format: str = "PNG"                   # PNG | JPEG | etc.
    image_path: str = ""                  # path to persisted cache file


# ══════════════════════════════════════════════════════════════
#  CHUNKING TYPES
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata carried by every structured chunk."""
    source_pdf: str
    page_number: int
    section_hierarchy: tuple = field(default_factory=tuple)  # ("H1 title", "H2 sub")
    element_type: str = "paragraph"
    city: str = ""
    scene_type: str = ""
    crowd_level: str = ""
    lighting: str = ""
    emotion_tags: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class StructuredChunk:
    """A layout-aware chunk ready for embedding."""
    content: str
    metadata: ChunkMetadata
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    chunk_index: int = 0
    token_count: int = 0


# ══════════════════════════════════════════════════════════════
#  INDEXING TYPES
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ImageNodeData:
    """
    Complete image node stored in the image index.
    image_vector and caption_vector are stored as tuples (immutable).
    """
    image_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image_vector: tuple = field(default_factory=tuple)     # CLIP 512-dim
    caption_vector: tuple = field(default_factory=tuple)   # BGE 768-dim (optional)
    caption: str = ""
    city: str = ""
    scene_type: str = ""
    crowd_level: str = ""
    lighting: str = ""
    emotion_tags: tuple = field(default_factory=tuple)
    source_pdf: str = ""
    page_number: int = 0
    image_path: str = ""                  # path to cached image file


# ══════════════════════════════════════════════════════════════
#  QUERY TYPES
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ProcessedQuery:
    """
    A user query after metadata extraction.
    Kept simple — embeddings computed on-demand inside query_processor.py.
    """
    text: str
    detected_metadata: Dict[str, Any] = field(default_factory=dict)
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ══════════════════════════════════════════════════════════════
#  RETRIEVAL TYPES
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RetrievalCandidate:
    """A single candidate node from any of the three indices."""
    node_id: str
    content: str
    similarity_score: float
    source_index: str                    # "text" | "image" | "caption"
    node_type: str                       # "text" | "image"
    rerank_score: float = 0.0            # populated by reranker
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "RetrievalCandidate") -> bool:
        return self.similarity_score < other.similarity_score


# ══════════════════════════════════════════════════════════════
#  OUTPUT TYPES
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RAGContext:
    """
    Final assembled context returned by pipeline.query().
    Ready for consumption by planner agents.
    """
    query: str
    assembled_prompt: str
    retrieved_text_nodes: tuple = field(default_factory=tuple)    # tuple of RetrievalCandidate
    retrieved_image_nodes: tuple = field(default_factory=tuple)   # tuple of RetrievalCandidate
    image_paths: tuple = field(default_factory=tuple)             # str paths to image files
    source_pdfs: tuple = field(default_factory=tuple)             # str PDF filenames
    token_count: int = 0


@dataclass(frozen=True)
class RAGResult:
    """
    Typed return wrapper for ALL pipeline.py public methods.

    The pipeline NEVER raises. It always returns a RAGResult.
    Callers check `success` before using `data`.
    `error` is a human-readable message if success=False.
    """
    success: bool
    data: Optional[Any] = None      # RAGContext for query(); dict for ingest()
    error: Optional[str] = None
    error_code: Optional[str] = None  # e.g. "PARSING_ERROR", "EMPTY_QUERY"
