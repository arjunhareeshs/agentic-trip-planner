"""
protocols.py — Abstract base classes (protocols) for ALL module boundaries.

PURPOSE:
  Define swappable interfaces. To swap any component:
    1. Write a class implementing the protocol below
    2. Update default.yaml with the new class path
    3. Pipeline auto-loads it — zero other files change

RULES:
  • Every protocol method has full type annotations.
  • No implementation logic here — only interface definitions.
  • All concrete classes in submodules MUST inherit from these.
  • numpy is imported lazily (TYPE_CHECKING only) to avoid hard dep at import time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import typing

if TYPE_CHECKING:
    import numpy as np

from .rag_types import (
    ExtractedImage,
    ImageNodeData,
    ParsedDocument,
    ProcessedQuery,
    RAGContext,
    RetrievalCandidate,
    StructuredChunk,
)


# ══════════════════════════════════════════════════════════════
#  PARSING PROTOCOLS
# ══════════════════════════════════════════════════════════════

class ParserProtocol(ABC):
    """
    Contract for PDF → structured document converters.
    Default implementation: DoclingParser
    """

    @abstractmethod
    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a single PDF file into structured document elements.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            ParsedDocument with ordered DocumentElement list.

        Raises:
            ParsingError: If the file is corrupt, unreadable, or password-protected.
        """
        ...

    @abstractmethod
    def is_supported(self, pdf_path: str) -> bool:
        """Return True if this parser can handle the given file."""
        ...


class ImageExtractorProtocol(ABC):
    """
    Contract for image extraction from PDFs.
    Default implementation: FitzImageExtractor
    """

    @abstractmethod
    def extract(self, pdf_path: str) -> List[ExtractedImage]:
        """
        Extract all embedded images from a PDF.

        Returns:
            List of ExtractedImage. Empty list if no images found (not an error).

        Raises:
            ParsingError: Only on file-level failures (file not found, corrupt).
        """
        ...


# ══════════════════════════════════════════════════════════════
#  CHUNKING PROTOCOL
# ══════════════════════════════════════════════════════════════

class ChunkerProtocol(ABC):
    """
    Contract for document → chunk converters.
    Default implementation: StructureChunker
    """

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> List[StructuredChunk]:
        """
        Split a ParsedDocument into layout-aware chunks.

        Returns:
            Ordered list of StructuredChunk. Never empty (raises if no chunks).

        Raises:
            ChunkingError: If document has zero parseable elements.
        """
        ...


# ══════════════════════════════════════════════════════════════
#  EMBEDDING PROTOCOLS
# ══════════════════════════════════════════════════════════════

class TextEmbedderProtocol(ABC):
    """
    Contract for text → dense vector embedders.
    Default implementation: BGETextEmbedder (BAAI/bge-base-en-v1.5)
    Output dimension: 768
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of output vectors."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> "np.ndarray":
        """
        Embed a list of strings.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), L2-normalized.

        Raises:
            EmbeddingError: On model failure or dimension mismatch.
        """
        ...

    def embed_single(self, text: str) -> "np.ndarray":
        """Convenience wrapper — embed one string."""
        return self.embed_batch([text])[0]


class ImageEmbedderProtocol(ABC):
    """
    Contract for image/text → CLIP-space embedders.
    Default implementation: CLIPImageEmbedder (openai/clip-vit-base-patch32)
    Output dimension: 512
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of output vectors."""
        ...

    @abstractmethod
    def embed_image(self, image: Any) -> "np.ndarray":
        """
        Embed a PIL.Image into CLIP space.

        Returns:
            np.ndarray of shape (512,), L2-normalized.

        Raises:
            EmbeddingError: On corrupt image or model failure.
        """
        ...

    @abstractmethod
    def embed_text(self, text: str) -> "np.ndarray":
        """
        Embed a text string into CLIP space (for caption alignment).

        Returns:
            np.ndarray of shape (512,), L2-normalized.
        """
        ...


# ══════════════════════════════════════════════════════════════
#  INDEXING PROTOCOL
# ══════════════════════════════════════════════════════════════

class IndexProtocol(ABC):
    """
    Contract for all three vector indices (text, image, caption).
    Shared interface means the retriever treats all three identically.
    """

    @abstractmethod
    def build(self, nodes: List[Any]) -> None:
        """
        Build the index from a list of nodes.

        Raises:
            IndexingError: On persistence failure or dimension mismatch.
        """
        ...

    @abstractmethod
    def add(self, nodes: List[Any]) -> None:
        """Incrementally add nodes to an existing index."""
        ...

    @abstractmethod
    def query(
        self,
        embedding: "np.ndarray",
        filters: Optional[Any],
        top_k: int,
    ) -> List[RetrievalCandidate]:
        """
        Vector similarity search.

        Returns:
            List of RetrievalCandidate sorted by similarity descending.
            Empty list if no results match (never raises on empty).
        """
        ...

    @abstractmethod
    def persist(self) -> None:
        """Save index to disk at configured path."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load index from disk. Call before querying a persisted index."""
        ...

    @abstractmethod
    def is_empty(self) -> bool:
        """True if the index has no nodes."""
        ...


# ══════════════════════════════════════════════════════════════
#  RANKING PROTOCOL
# ══════════════════════════════════════════════════════════════

class RankerProtocol(ABC):
    """Contract for cross-encoder reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int = 5,
    ) -> List[RetrievalCandidate]:
        """
        Rerank candidates using cross-encoder scoring.

        Fallback guarantee: If the model fails, returns candidates sorted
        by original similarity_score (never returns empty if input non-empty).

        Returns:
            Up to top_k RetrievalCandidates with updated rerank_score.
        """
        ...


# ══════════════════════════════════════════════════════════════
#  ASSEMBLY PROTOCOL
# ══════════════════════════════════════════════════════════════

class AssemblerProtocol(ABC):
    """Contract for final context assembly."""

    @abstractmethod
    def assemble(
        self,
        query: str,
        reranked_candidates: List[RetrievalCandidate],
        all_text_chunks: Optional[List[StructuredChunk]] = None,
    ) -> RAGContext:
        """
        Combine ranked results into a final RAGContext.
        Never raises — assembles whatever is available.
        """
        ...


# ══════════════════════════════════════════════════════════════
#  QUERY PROCESSING PROTOCOLS
# ══════════════════════════════════════════════════════════════

class QueryProcessorProtocol(ABC):
    """Contract for the query processing orchestrator."""

    @abstractmethod
    def process(self, query: "ProcessedQuery") -> List[RetrievalCandidate]:
        """
        Process a user query through embedding, retrieval, and deduplication.

        Returns:
            Deduplicated, scored list of RetrievalCandidates.
        """
        ...


class RetrieverProtocol(ABC):
    """Contract for multi-index parallel retrieval."""

    @abstractmethod
    def retrieve(
        self,
        text_embedding: Any,
        clip_embedding: Any,
        filters: Optional[Any] = None,
        top_k_map: Optional[Dict[str, int]] = None,
    ) -> List[RetrievalCandidate]:
        """
        Run parallel retrieval across indices.

        Returns:
            Merged list of RetrievalCandidates from all indices.
        """
        ...


class FilterProtocol(ABC):
    """Contract for metadata filter builders."""

    @abstractmethod
    def build_filters(self, metadata: Dict[str, Any]) -> Optional[Any]:
        """
        Convert raw metadata dict into a vector-store filter object.

        Returns:
            Filter object or None if no filters apply.
        """
        ...
