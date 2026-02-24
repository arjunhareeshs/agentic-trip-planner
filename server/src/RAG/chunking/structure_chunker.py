"""
structure_chunker.py — Layout-aware, structure-driven document chunker.

Implements ChunkerProtocol. Creates chunks based on document structure,
NOT naive token counting:
  • heading + child paragraphs → one chunk (section chunk)
  • table → standalone chunk (never split mid-table)
  • bullet/numbered list → standalone chunk
  • pricing_block → standalone chunk
  • image_caption → standalone chunk
  • Overflow sections (> max_chunk_tokens) → split at paragraph boundaries only
  • Overlap added at section boundaries for context continuity

ISOLATION: Imports ONLY from types, protocols, utils. No other RAG modules.

USAGE:
    chunker = StructureChunker(chunking_config)
    chunks = chunker.chunk(parsed_document)
    for chunk in chunks:
        print(chunk.element_type, chunk.token_count, chunk.content[:60])
"""

from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple

from ..protocols import ChunkerProtocol
from ..rag_types import ChunkMetadata, DocumentElement, ParsedDocument, StructuredChunk
from ..utils.exceptions import ChunkingError
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Element types that are always kept as standalone (never merged)
_ALWAYS_STANDALONE = frozenset(["table", "list", "pricing_block", "image_caption", "map"])

# Element types that can be merged into section chunks
_MERGEABLE = frozenset(["paragraph", "unknown"])


class StructureChunker(ChunkerProtocol):
    """
    Structure-aware chunker that respects document layout.

    Args:
        chunking_config: The chunking sub-section of RAGSettings.
    """

    def __init__(self, chunking_config=None):
        self._config = chunking_config
        self._max_tokens = getattr(chunking_config, "max_chunk_tokens", 512)
        self._overlap_tokens = getattr(chunking_config, "overlap_tokens", 50)
        self._min_tokens = getattr(chunking_config, "min_chunk_tokens", 30)

    # ── Public API ────────────────────────────────────────────

    def chunk(self, document: ParsedDocument) -> List[StructuredChunk]:
        """
        Chunk a ParsedDocument into structure-aware StructuredChunk list.

        Raises:
            ChunkingError: If document has zero parseable elements.
        """
        t0 = time.perf_counter()

        if not document.elements:
            raise ChunkingError(
                "ParsedDocument has zero elements — cannot chunk.",
                context={"source_pdf": document.source_pdf},
            )

        chunks = self._build_chunks(document)

        if not chunks:
            raise ChunkingError(
                "Chunking produced zero chunks — all elements were filtered out.",
                context={"source_pdf": document.source_pdf, "total_elements": len(document.elements)},
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Chunked %d elements -> %d chunks from %s in %.2fs",
            len(document.elements), len(chunks),
            document.source_pdf.split("/")[-1] if "/" in document.source_pdf else document.source_pdf,
            elapsed,
        )

        return chunks

    # ── Core chunking logic ───────────────────────────────────

    def _build_chunks(self, document: ParsedDocument) -> List[StructuredChunk]:
        """Drive the main chunking pass over all document elements."""
        chunks: List[StructuredChunk] = []
        chunk_index = 0

        # Track current section heading stack for hierarchy metadata
        heading_stack: List[Tuple[int, str]] = []   # [(level, text), ...]

        # Accumulator for the current section (heading + mergeable paragraphs)
        current_section: List[DocumentElement] = []
        current_section_page: int = 1

        elements = list(document.elements)

        for i, element in enumerate(elements):
            elem_type = element.element_type

            # ── Standalone elements: flush current section, emit standalone ──
            if elem_type in _ALWAYS_STANDALONE:
                if current_section:
                    new_chunks = self._emit_section_chunks(
                        current_section, heading_stack, document.source_pdf, chunk_index
                    )
                    chunks.extend(new_chunks)
                    chunk_index += len(new_chunks)
                    current_section = []

                standalone = self._make_standalone_chunk(
                    element, heading_stack, document.source_pdf, chunk_index
                )
                if standalone:
                    chunks.append(standalone)
                    chunk_index += 1

            # ── Heading: flush current section, update heading stack ──
            elif elem_type == "heading":
                if current_section:
                    new_chunks = self._emit_section_chunks(
                        current_section, heading_stack, document.source_pdf, chunk_index
                    )
                    chunks.extend(new_chunks)
                    chunk_index += len(new_chunks)
                    current_section = []

                # Update heading stack
                heading_stack = self._update_heading_stack(heading_stack, element)
                # Start new section with this heading
                current_section = [element]
                current_section_page = element.page_number

            # ── Mergeable elements: add to current section ──
            elif elem_type in _MERGEABLE:
                current_section.append(element)

            # ── Paragraph: add to current section ──
            elif elem_type == "paragraph":
                current_section.append(element)

        # Flush any remaining section
        if current_section:
            new_chunks = self._emit_section_chunks(
                current_section, heading_stack, document.source_pdf, chunk_index
            )
            chunks.extend(new_chunks)

        return chunks

    # ── Section emission ──────────────────────────────────────

    def _emit_section_chunks(
        self,
        section: List[DocumentElement],
        heading_stack: List[Tuple[int, str]],
        source_pdf: str,
        base_index: int,
    ) -> List[StructuredChunk]:
        """
        Convert a section (heading + paragraphs) into one or more chunks.
        Splits at paragraph boundaries if section exceeds max_chunk_tokens.
        """
        full_text = "\n\n".join(e.content for e in section if e.content.strip())
        full_text = full_text.strip()

        if not full_text or self._count_tokens(full_text) < self._min_tokens:
            return []

        page_number = section[0].page_number if section else 1
        section_hierarchy = tuple(text for _, text in heading_stack)
        elem_type = self._primary_element_type(section)

        # Build metadata (no city/scene yet — added by ingestion stage)
        metadata = ChunkMetadata(
            source_pdf=source_pdf,
            page_number=page_number,
            section_hierarchy=section_hierarchy,
            element_type=elem_type,
        )

        if self._count_tokens(full_text) <= self._max_tokens:
            # Single chunk — fits within limit
            return [StructuredChunk(
                content=full_text,
                metadata=metadata,
                chunk_index=base_index,
                token_count=self._count_tokens(full_text),
            )]

        # Overflow — split at paragraph boundaries
        return self._split_at_paragraphs(full_text, metadata, base_index)

    def _split_at_paragraphs(
        self,
        text: str,
        metadata: ChunkMetadata,
        base_index: int,
    ) -> List[StructuredChunk]:
        """Split a long text at paragraph boundaries, adding overlap."""
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        chunks: List[StructuredChunk] = []
        current_parts: List[str] = []
        current_tokens = 0
        chunk_i = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if current_tokens + para_tokens > self._max_tokens and current_parts:
                # Emit current accumulation
                chunk_text = "\n\n".join(current_parts)
                chunks.append(StructuredChunk(
                    content=chunk_text,
                    metadata=metadata,
                    chunk_index=base_index + chunk_i,
                    token_count=current_tokens,
                ))
                chunk_i += 1

                # Overlap: keep last N tokens of current chunk
                overlap_text = self._get_overlap_suffix(chunk_text)
                current_parts = [overlap_text, para] if overlap_text else [para]
                current_tokens = self._count_tokens("\n\n".join(current_parts))
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            if self._count_tokens(chunk_text) >= self._min_tokens:
                chunks.append(StructuredChunk(
                    content=chunk_text,
                    metadata=metadata,
                    chunk_index=base_index + chunk_i,
                    token_count=self._count_tokens(chunk_text),
                ))

        return chunks

    # ── Standalone chunk creation ─────────────────────────────

    def _make_standalone_chunk(
        self,
        element: DocumentElement,
        heading_stack: List[Tuple[int, str]],
        source_pdf: str,
        chunk_index: int,
    ) -> Optional[StructuredChunk]:
        """Create a standalone chunk for a table/list/pricing/caption element."""
        content = element.content.strip()

        # For tables, prefer structured text representation
        if element.element_type == "table" and element.table_data:
            content = self._table_to_text(element.table_data)

        if not content or self._count_tokens(content) < self._min_tokens:
            return None

        section_hierarchy = tuple(text for _, text in heading_stack)
        metadata = ChunkMetadata(
            source_pdf=source_pdf,
            page_number=element.page_number,
            section_hierarchy=section_hierarchy,
            element_type=element.element_type,
        )

        return StructuredChunk(
            content=content,
            metadata=metadata,
            chunk_index=chunk_index,
            token_count=self._count_tokens(content),
        )

    # ── Utilities ─────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """
        Approximate token count. Uses whitespace splitting as proxy
        (actual tokenizer would be slower; ~1.3 words per token on average).
        """
        return max(1, len(text.split()))

    def _get_overlap_suffix(self, text: str) -> str:
        """Return the last `overlap_tokens` words of text for overlap."""
        words = text.split()
        if len(words) <= self._overlap_tokens:
            return text
        return " ".join(words[-self._overlap_tokens:])

    def _update_heading_stack(
        self,
        stack: List[Tuple[int, str]],
        heading: DocumentElement,
    ) -> List[Tuple[int, str]]:
        """
        Update heading hierarchy stack for the new heading.
        Pops any headings at same or deeper level before pushing.
        """
        level = heading.level or 1
        new_stack = [(lvl, txt) for lvl, txt in stack if lvl < level]
        new_stack.append((level, heading.content.strip()))
        return new_stack

    def _primary_element_type(self, section: List[DocumentElement]) -> str:
        """Determine the dominant element type in a section."""
        if not section:
            return "paragraph"
        first = section[0].element_type
        if first == "heading":
            return "section"
        return first

    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table cells to a readable pipe-delimited text format."""
        if not table_data:
            return ""
        rows = []
        for i, row in enumerate(table_data):
            rows.append(" | ".join(cell.strip() for cell in row))
            if i == 0:
                rows.append("-" * max(len(rows[0]), 10))  # header separator
        return "\n".join(rows)
