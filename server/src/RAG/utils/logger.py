"""
logger.py — Structured, module-scoped logging for the RAG pipeline.

USAGE:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Parsing %s", pdf_path)

FORMAT:
    [2026-02-23 10:00:00] [INFO ] [rag.parsing.docling_parser] Parsing brochure.pdf

RULES:
  • Call get_logger(__name__) at module top level.
  • Never use print() statements — all output goes through loggers.
  • Log level is set once from config and propagated to all rag.* children.
  • This module has ZERO dependencies on other RAG modules (no circular imports).
"""

import io
import logging
import sys
from typing import Optional

# Root logger name — all submodule loggers are children of this
_RAG_ROOT_LOGGER = "rag"
_INITIALIZED = False


def _build_formatter() -> logging.Formatter:
    """Create the canonical log formatter."""
    return logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-5s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def initialize_logging(level: str = "INFO") -> None:
    """
    Configure the root RAG logger. Call once at pipeline startup.
    Safe to call multiple times — only initializes once.

    Args:
        level: Logging level string from config (DEBUG/INFO/WARNING/ERROR).
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger(_RAG_ROOT_LOGGER)
    root_logger.setLevel(numeric_level)

    # Console handler — always present (UTF-8 safe on Windows)
    stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(_build_formatter())
    root_logger.addHandler(console_handler)

    # Prevent log propagation to root Python logger
    root_logger.propagate = False

    _INITIALIZED = True


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a module-scoped logger.

    Args:
        module_name: Pass __name__ — e.g. "rag.parsing.docling_parser"
                     If the name doesn't start with "rag.", it is prefixed.

    Returns:
        A Logger instance that inherits level from the root RAG logger.
    """
    if not module_name.startswith(_RAG_ROOT_LOGGER):
        # Strip package path and make it a child of rag.*
        short = module_name.split(".")[-1]
        logger_name = f"{_RAG_ROOT_LOGGER}.{short}"
    else:
        logger_name = module_name

    return logging.getLogger(logger_name)


def set_level(level: str) -> None:
    """Dynamically change the log level at runtime."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger(_RAG_ROOT_LOGGER).setLevel(numeric_level)
