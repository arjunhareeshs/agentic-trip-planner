"""
context_assembler.py — Final context assembly from reranked candidates.

Implements AssemblerProtocol.
  Combines reranked image nodes, their supporting text context, and
  optional knowledge graph relations into a RAGContext object within
  the configured token budget.

OUTPUT STRUCTURE:
  • retrieved_text_nodes: top text chunks
  • retrieved_image_nodes: top image nodes (with image_path for rendering)
  • supporting_text: adjacent text paragraphs near top images (by page proximity)
  • assembled_prompt: the ready-to-use context block for LLM generation
  • token_count: estimated tokens in assembled_prompt

ISOLATION: Imports ONLY from protocols, types, utils. No other RAG submodules.

USAGE:
    assembler = ContextAssembler(retrieval_config)
    context = assembler.assemble(query, reranked_candidates, text_chunks)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set

from ..protocols import AssemblerProtocol
from ..rag_types import ImageNodeData, RAGContext, RetrievalCandidate, StructuredChunk
from ..utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_ESTIMATE_FACTOR = 1.3   # chars per token (rough average)


class ContextAssembler(AssemblerProtocol):
    """
    Context assembler — merges text and image nodes into RAGContext.

    Args:
        retrieval_config: RetrievalConfig from RAGSettings.
    """

    def __init__(self, retrieval_config=None):
        self._config = retrieval_config
        self._token_budget = getattr(retrieval_config, "context_token_budget", 4096)

    # ── AssemblerProtocol ─────────────────────────────────────

    def assemble(
        self,
        query: str,
        reranked_candidates: List[RetrievalCandidate],
        all_text_chunks: Optional[List[StructuredChunk]] = None,
    ) -> RAGContext:
        """
        Build the final RAGContext from reranked candidates.

        Args:
            query: Original query string.
            reranked_candidates: Sorted list of RetrievalCandidates (highest score first).
            all_text_chunks: All ingested text chunks (for supporting text lookup).
                             Optional. Can be None if not available.

        Returns:
            RAGContext — typed, immutable object ready for LLM consumption.
        """
        t0 = time.perf_counter()

        # Partition reranked candidates by type
        text_candidates = [c for c in reranked_candidates if c.node_type == "text"]
        image_candidates = self._deduplicate_by_image_id(
            [c for c in reranked_candidates if c.node_type == "image"]
        )

        # Build text context sections
        text_sections = self._build_text_sections(text_candidates)

        # Build image sections (with page-neighbor text support)
        image_sections = self._build_image_sections(image_candidates, all_text_chunks)

        # Assemble the prompt string
        assembled_prompt = self._assemble_prompt(
            query, text_sections, image_sections
        )
        token_estimate = int(len(assembled_prompt) / _TOKEN_ESTIMATE_FACTOR)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Context assembled: %d text + %d images, ~%d tokens in %.2fs",
            len(text_sections),
            len(image_sections),
            token_estimate,
            elapsed,
        )

        # Extract image paths for retrieval
        image_paths = [
            c.metadata.get("image_path", "") for c in image_candidates
            if c.metadata.get("image_path")
        ]

        return RAGContext(
            query=query,
            retrieved_text_nodes=tuple(text_candidates),
            retrieved_image_nodes=tuple(image_candidates),
            assembled_prompt=assembled_prompt,
            token_count=token_estimate,
            image_paths=tuple(image_paths),
            source_pdfs=tuple(self._collect_sources(reranked_candidates)),
        )

    # ── Text section building ─────────────────────────────────

    def _build_text_sections(
        self, candidates: List[RetrievalCandidate]
    ) -> List[Dict]:
        """Convert text candidates into formatted section dicts."""
        sections = []
        remaining_budget = self._token_budget

        for c in candidates:
            content = c.content.strip()
            estimated_tokens = int(len(content) / _TOKEN_ESTIMATE_FACTOR)
            if estimated_tokens > remaining_budget:
                content = self._truncate_to_budget(content, remaining_budget)
                estimated_tokens = int(len(content) / _TOKEN_ESTIMATE_FACTOR)

            if not content:
                break

            sections.append({
                "content": content,
                "source_pdf": c.metadata.get("source_pdf", ""),
                "page": c.metadata.get("page_number", 0),
                "score": c.rerank_score or c.similarity_score,
            })
            remaining_budget -= estimated_tokens

            if remaining_budget <= 0:
                break

        return sections

    # ── Image section building ────────────────────────────────

    def _build_image_sections(
        self,
        candidates: List[RetrievalCandidate],
        all_text_chunks: Optional[List[StructuredChunk]],
    ) -> List[Dict]:
        """
        Build image sections, attaching nearby text context by page proximity.
        """
        sections = []
        for c in candidates:
            image_page = c.metadata.get("page_number", 0)
            source_pdf = c.metadata.get("source_pdf", "")

            # Find text chunks on the same or adjacent pages
            supporting = ""
            if all_text_chunks:
                nearby = [
                    chunk.content
                    for chunk in all_text_chunks
                    if (
                        chunk.metadata.source_pdf == source_pdf
                        and abs(chunk.metadata.page_number - image_page) <= 1
                    )
                ]
                supporting = " ".join(nearby[:2])[:300] if nearby else ""

            sections.append({
                "image_id": c.metadata.get("image_id", c.node_id),
                "caption": c.metadata.get("caption", ""),
                "image_path": c.metadata.get("image_path", ""),
                "page": image_page,
                "source_pdf": source_pdf,
                "supporting_text": supporting,
                "score": c.rerank_score or c.similarity_score,
            })

        return sections

    # ── Prompt assembly ───────────────────────────────────────

    def _assemble_prompt(
        self,
        query: str,
        text_sections: List[Dict],
        image_sections: List[Dict],
    ) -> str:
        """Format the assembled context as a structured prompt block."""
        parts = [
            f"## Query\n{query}\n",
            "## Retrieved Context\n",
        ]

        for i, section in enumerate(text_sections, start=1):
            source = section.get("source_pdf", "").split("/")[-1] or "unknown"
            page = section.get("page", "?")
            parts.append(
                f"### Text Source {i} [{source}, p.{page}]\n{section['content']}\n"
            )

        if image_sections:
            parts.append("## Relevant Images\n")
            for i, section in enumerate(image_sections, start=1):
                caption = section.get("caption") or "(no caption)"
                source = section.get("source_pdf", "").split("/")[-1] or "unknown"
                page = section.get("page", "?")
                supporting = section.get("supporting_text", "")
                parts.append(
                    f"### Image {i} [{source}, p.{page}]\n"
                    f"**Caption**: {caption}\n"
                )
                if supporting:
                    parts.append(f"**Nearby text**: {supporting}\n")

        return "\n".join(parts)

    # ── Utilities ─────────────────────────────────────────────

    def _deduplicate_by_image_id(
        self, candidates: List[RetrievalCandidate]
    ) -> List[RetrievalCandidate]:
        """Deduplicate image candidates by image_id, keeping highest score."""
        seen: Dict[str, RetrievalCandidate] = {}
        for c in candidates:
            image_id = c.metadata.get("image_id", c.node_id)
            if image_id not in seen or c.similarity_score > seen[image_id].similarity_score:
                seen[image_id] = c
        return list(seen.values())

    def _truncate_to_budget(self, text: str, token_budget: int) -> str:
        """Truncate text to approximately token_budget tokens."""
        char_limit = int(token_budget * _TOKEN_ESTIMATE_FACTOR)
        return text[:char_limit].rsplit(" ", 1)[0]  # clean word boundary

    def _collect_sources(self, candidates: List[RetrievalCandidate]) -> Set[str]:
        """Collect unique source PDF filenames."""
        return {
            c.metadata.get("source_pdf", "").split("/")[-1]
            for c in candidates
            if c.metadata.get("source_pdf")
        }
