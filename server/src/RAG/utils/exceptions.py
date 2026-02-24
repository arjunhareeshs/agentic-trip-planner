"""
exceptions.py — Custom exception hierarchy for the RAG pipeline.

RULES:
  • Every exception in this module inherits from RAGError.
  • Library exceptions (OSError, ValueError, etc.) are NEVER raised
    directly from module functions. They must be caught and wrapped
    into an appropriate RAGError subclass with a descriptive message.
  • Only RAGError subtypes cross module boundaries.
  • pipeline.py catches ALL RAGError subtypes at its error boundary.
"""


class RAGError(Exception):
    """
    Base class for all RAG pipeline exceptions.
    Always includes a human-readable message and optional context dict.
    """

    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.error_code = self.__class__.__name__.upper().replace("ERROR", "_ERROR")

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{ctx}]"
        return self.message


class ConfigError(RAGError):
    """
    Raised when configuration is missing, invalid, or fails validation.

    Examples:
      - Required YAML key missing
      - Invalid device string (not auto/cpu/cuda/mps)
      - Model path points to non-existent directory
    """
    pass


class ValidationError(RAGError):
    """
    Raised by validators.py when input data fails validation checks.

    Examples:
      - PDF file not found or not readable
      - Query string is empty or exceeds token limit
      - Embedding vector has wrong dimension
    """
    pass


class ParsingError(RAGError):
    """
    Raised during PDF parsing (docling or fitz stage).

    Examples:
      - PDF is corrupt or truncated
      - PDF is password-protected
      - PDF has zero extractable text pages
    """
    pass


class ChunkingError(RAGError):
    """
    Raised during structure-aware chunking.

    Examples:
      - ParsedDocument has zero parseable elements
      - Chunking produces zero chunks (all filtered out)
    """
    pass


class EmbeddingError(RAGError):
    """
    Raised during model loading or embedding computation.

    Examples:
      - Model weights not found and download fails
      - CUDA OOM during batch embedding
      - Output dimension does not match config
    """
    pass


class IndexingError(RAGError):
    """
    Raised during index build, update, or persistence.

    Examples:
      - Index persistence directory not writable
      - Embedding dimension mismatch between stored and new nodes
      - Index file corrupt on load
    """
    pass


class QueryError(RAGError):
    """
    Raised during query processing or routing.

    Examples:
      - Empty query string after stripping whitespace
      - Query exceeds maximum token length
      - Query engine misconfiguration
    """
    pass


class RankingError(RAGError):
    """
    Raised by the cross-encoder reranker.

    NOTE: RankingError triggers fallback to similarity-score ordering,
    NOT a pipeline failure. The pipeline catches this specifically.
    """
    pass


class RetrievalError(RAGError):
    """
    Raised when retrieval from an index fails catastrophically.

    NOTE: Individual index failures in parallel retrieval are logged
    and skipped, not raised. This is only raised if ALL indices fail.
    """
    pass
