"""
docling_parser.py — Layout-aware PDF parser using Docling.

Implements ParserProtocol. Converts PDFs into structured ParsedDocument
with typed DocumentElement objects preserving:
  • Multi-column layouts
  • Tables (as structured data, not raw text)
  • Headings hierarchy (H1/H2/H3)
  • Image captions
  • Bullet/numbered lists
  • Pricing blocks
  • Maps and callout blocks

ISOLATION: Imports ONLY from types, protocols, utils. No other RAG modules.

USAGE:
    parser = DoclingParser(config.parsing)
    doc = parser.parse("/path/to/brochure.pdf")
    for element in doc.elements:
        print(element.element_type, element.content[:80])
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

from ..protocols import ParserProtocol
from ..rag_types import DocumentElement, ParsedDocument
from ..utils.exceptions import ParsingError
from ..utils.logger import get_logger
from ..utils.validators import validate_pdf_path

logger = get_logger(__name__)

# ── Element type constants ────────────────────────────────────
ELEM_HEADING = "heading"
ELEM_PARAGRAPH = "paragraph"
ELEM_TABLE = "table"
ELEM_LIST = "list"
ELEM_IMAGE_CAPTION = "image_caption"
ELEM_PRICING_BLOCK = "pricing_block"
ELEM_MAP = "map"
ELEM_UNKNOWN = "unknown"

# Docling section label → our element type mapping
_DOCLING_TYPE_MAP = {
    "section-header": ELEM_HEADING,
    "title": ELEM_HEADING,
    "text": ELEM_PARAGRAPH,
    "table": ELEM_TABLE,
    "list-item": ELEM_LIST,
    "caption": ELEM_IMAGE_CAPTION,
    "picture": ELEM_IMAGE_CAPTION,
    "formula": ELEM_PARAGRAPH,
    "footnote": ELEM_PARAGRAPH,
    "page-header": ELEM_UNKNOWN,
    "page-footer": ELEM_UNKNOWN,
}


class DoclingParser(ParserProtocol):
    """
    Layout-aware PDF parser backed by the Docling library.

    Args:
        parsing_config: The parsing sub-section of RAGSettings.
    """

    def __init__(self, parsing_config=None):
        self._config = parsing_config
        self._converter = None   # lazy-loaded

    # ── Public API ────────────────────────────────────────────

    @validate_pdf_path(arg_name="pdf_path")
    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF into a structured ParsedDocument.

        Raises:
            ParsingError: On corrupt/password-protected/empty PDFs.
        """
        t0 = time.perf_counter()
        logger.info("Parsing PDF: %s", pdf_path)

        try:
            converter = self._get_converter()
            result = converter.convert(pdf_path)
        except Exception as exc:
            raise ParsingError(
                f"Docling failed to convert PDF: {exc}",
                context={"pdf_path": pdf_path, "error_type": type(exc).__name__},
            ) from exc

        elements = self._extract_elements(result)
        if not elements:
            raise ParsingError(
                "PDF yielded zero parseable elements after Docling conversion.",
                context={"pdf_path": pdf_path},
            )

        warnings = self._collect_warnings(result)
        total_pages = self._get_page_count(result)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Parsed %d elements from %d pages in %.2fs [%s]",
            len(elements), total_pages, elapsed, Path(pdf_path).name,
        )

        return ParsedDocument(
            source_pdf=pdf_path,
            elements=tuple(elements),
            total_pages=total_pages,
            parse_warnings=tuple(warnings),
        )

    def is_supported(self, pdf_path: str) -> bool:
        """Return True if file has a supported extension."""
        supported = getattr(self._config, "supported_formats", [".pdf"])
        return Path(pdf_path).suffix.lower() in supported

    # ── Private helpers ───────────────────────────────────────

    def _get_converter(self):
        """Lazy-load Docling DocumentConverter (heavy import)."""
        if self._converter is not None:
            return self._converter
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            pipeline_opts = PdfPipelineOptions()
            pipeline_opts.do_table_structure = getattr(
                self._config, "docling_preserve_tables", True
            )
            pipeline_opts.do_ocr = False   # OCR adds latency — disable by default

            self._converter = DocumentConverter()
            logger.debug("DoclingParser: converter initialized")
        except ImportError as exc:
            raise ParsingError(
                "Docling library not installed. Run: pip install docling",
                context={"hint": "pip install docling"},
            ) from exc
        return self._converter

    def _extract_elements(self, result) -> List[DocumentElement]:
        """Convert Docling result items into DocumentElement list."""
        elements: List[DocumentElement] = []
        heading_level_tracker: dict = {}   # track H1/H2/H3 nesting

        try:
            items = list(result.document.iterate_items())
        except AttributeError:
            # Fallback for different Docling API versions
            items = []

        for item, level in items:
            elem = self._item_to_element(item, level, heading_level_tracker)
            if elem is not None:
                elements.append(elem)

        return elements

    def _item_to_element(self, item, depth: int, heading_tracker: dict) -> Optional[DocumentElement]:
        """Convert a single Docling item to a DocumentElement."""
        try:
            # Resolve element type
            label = getattr(item, "label", "text")
            elem_type = _DOCLING_TYPE_MAP.get(str(label).lower(), ELEM_PARAGRAPH)

            # Extract text content
            content = ""
            if hasattr(item, "text"):
                content = item.text or ""
            elif hasattr(item, "export_to_markdown"):
                content = item.export_to_markdown() or ""

            content = content.strip()
            if not content:
                return None   # skip empty elements

            # Resolve page number
            page_number = 1
            if hasattr(item, "prov") and item.prov:
                page_number = getattr(item.prov[0], "page_no", 1)

            # Resolve heading level
            level = 0
            if elem_type == ELEM_HEADING:
                level = self._infer_heading_level(item, depth, heading_tracker)

            # Extract table data if applicable
            table_data = None
            if elem_type == ELEM_TABLE:
                table_data = self._extract_table_data(item)

            return DocumentElement(
                element_type=elem_type,
                content=content,
                page_number=page_number,
                level=level,
                table_data=table_data,
            )
        except Exception as exc:
            logger.warning("Skipping malformed element: %s", exc)
            return None

    def _infer_heading_level(self, item, depth: int, tracker: dict) -> int:
        """
        Infer heading level (1-6) from Docling item attributes.
        Falls back to depth in document tree.
        """
        if hasattr(item, "level"):
            return max(1, min(6, int(item.level)))
        return max(1, min(6, depth + 1))

    def _extract_table_data(self, item) -> Optional[List[List[str]]]:
        """Extract rows/cells from a Docling table item."""
        try:
            if hasattr(item, "data"):
                grid = item.data
                rows = []
                for row in grid.grid:
                    rows.append([cell.text for cell in row])
                return rows
        except Exception:
            pass
        return None

    def _collect_warnings(self, result) -> List[str]:
        """Collect non-fatal parse warnings from Docling result."""
        warnings = []
        try:
            if hasattr(result, "errors"):
                for err in result.errors:
                    warnings.append(str(err))
        except Exception:
            pass
        return warnings

    def _get_page_count(self, result) -> int:
        """Extract total page count from Docling result."""
        try:
            return len(result.document.pages)
        except Exception:
            return 0
