"""
validators.py — Input validation decorators for RAG module entry points.

PURPOSE:
  Apply these decorators to module entry-point functions so bad data
  is caught immediately at the boundary — not 5 layers deep in libraries.

USAGE:
    from utils.validators import validate_pdf_path, validate_query_string

    @validate_pdf_path(arg_name="pdf_path")
    def parse(self, pdf_path: str) -> ParsedDocument: ...

    @validate_query_string(arg_name="raw_query")
    def process(self, raw_query: str) -> ProcessedQuery: ...

RULES:
  • Validators only raise ValidationError — never raw Python exceptions.
  • They check the minimal viable constraints, not business logic.
  • Zero dependencies on other RAG modules except utils.exceptions.
"""

from __future__ import annotations

import functools
import os
from typing import Callable

from .exceptions import ValidationError


# ── PDF path validation ───────────────────────────────────────

def validate_pdf_path(arg_name: str = "pdf_path") -> Callable:
    """
    Decorator: validates that the given argument is a readable .pdf file.

    Checks:
      1. Argument is a non-empty string
      2. File exists on disk
      3. File extension is .pdf (case-insensitive)
      4. File is readable (not locked / permissions issue)
      5. File is not empty (0 bytes)

    Raises:
        ValidationError with descriptive message on any check failure.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Resolve the pdf_path value from positional or keyword args
            pdf_path = _resolve_arg(func, args, kwargs, arg_name)

            if not isinstance(pdf_path, str) or not pdf_path.strip():
                raise ValidationError(
                    f"'{arg_name}' must be a non-empty string.",
                    context={arg_name: repr(pdf_path)},
                )

            if not os.path.exists(pdf_path):
                raise ValidationError(
                    f"PDF file not found: {pdf_path}",
                    context={"path": pdf_path},
                )

            if not pdf_path.lower().endswith(".pdf"):
                raise ValidationError(
                    f"File is not a PDF (wrong extension): {pdf_path}",
                    context={"path": pdf_path},
                )

            if not os.access(pdf_path, os.R_OK):
                raise ValidationError(
                    f"PDF file is not readable (permissions?): {pdf_path}",
                    context={"path": pdf_path},
                )

            if os.path.getsize(pdf_path) == 0:
                raise ValidationError(
                    f"PDF file is empty (0 bytes): {pdf_path}",
                    context={"path": pdf_path},
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


# ── Query string validation ───────────────────────────────────

def validate_query_string(
    arg_name: str = "raw_query",
    max_chars: int = 2000,
) -> Callable:
    """
    Decorator: validates that the given argument is a usable query string.

    Checks:
      1. Argument is a string
      2. After stripping whitespace, length > 0
      3. Length does not exceed max_chars (configurable)

    Raises:
        ValidationError with descriptive message on any check failure.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            query = _resolve_arg(func, args, kwargs, arg_name)

            if not isinstance(query, str):
                raise ValidationError(
                    f"'{arg_name}' must be a string, "
                    f"got {type(query).__name__}.",
                    context={arg_name: repr(query)},
                )

            if not query.strip():
                raise ValidationError(
                    "Query string must not be empty or whitespace-only.",
                    context={arg_name: repr(query)},
                )

            if len(query) > max_chars:
                raise ValidationError(
                    f"Query string exceeds max length "
                    f"of {max_chars} characters "
                    f"(got {len(query)} chars). "
                    f"Truncate before calling.",
                    context={
                        "length": len(query),
                        "max": max_chars,
                    },
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


# ── Embedding dimension validation ────────────────────────────

def validate_embedding_dim(
    expected_dim: int,
    arg_name: str = "embedding",
) -> Callable:
    """
    Decorator: validates that an np.ndarray argument has the expected shape.

    Raises:
        ValidationError if dimension does not match.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import numpy as np
            embedding = _resolve_arg(func, args, kwargs, arg_name)

            if not isinstance(embedding, np.ndarray):
                raise ValidationError(
                    f"'{arg_name}' must be a numpy ndarray.",
                    context={"type": type(embedding).__name__},
                )

            actual_dim = embedding.shape[-1]
            if actual_dim != expected_dim:
                raise ValidationError(
                    f"Embedding dimension mismatch: expected {expected_dim}, "
                    f"got {actual_dim}.",
                    context={"expected": expected_dim, "actual": actual_dim},
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


# ── Chunk list validation ─────────────────────────────────────

def validate_chunks(arg_name: str = "chunks") -> Callable:
    """
    Decorator: validates that a chunk list is non-empty
    and has no empty content.

    Raises:
        ValidationError if list is empty or has
        chunks with empty content.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            chunks = _resolve_arg(func, args, kwargs, arg_name)

            if not chunks:
                raise ValidationError(
                    "Chunk list is empty — nothing to index.",
                    context={arg_name: "[]"},
                )

            empty_ids = [
                getattr(c, "chunk_id", str(i))
                for i, c in enumerate(chunks)
                if not getattr(
                    c, "content", ""
                ).strip()
            ]
            if empty_ids:
                raise ValidationError(
                    f"Found {len(empty_ids)} chunk(s) with empty content.",
                    context={"empty_chunk_ids": empty_ids[:5]},
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator


# ── Internal helper ───────────────────────────────────────────

def _resolve_arg(func: Callable, args: tuple, kwargs: dict, arg_name: str):
    """
    Resolve a named argument from either positional args or kwargs.
    Uses the function's signature to map positional to names.
    """
    import inspect
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    if arg_name in kwargs:
        return kwargs[arg_name]

    try:
        idx = param_names.index(arg_name)
        return args[idx]
    except (ValueError, IndexError):
        raise ValidationError(
            f"Validator could not find argument '{arg_name}' in function "
            f"'{func.__name__}'. Check decorator configuration.",
        )
