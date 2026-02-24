"""
image_extractor.py — PDF image extraction using fitz (PyMuPDF).

Implements ImageExtractorProtocol. Extracts embedded images from PDFs,
associates each image with its nearest caption via spatial proximity,
and saves images to a deterministic cache path.

ISOLATION: Imports ONLY from types, protocols, utils. No other RAG modules.

USAGE:
    extractor = FitzImageExtractor(parsing_config, cache_dir="./image_cache")
    images = extractor.extract("/path/to/brochure.pdf")
    for img in images:
        print(img.page_number, img.caption, img.image_path)
"""

from __future__ import annotations

import hashlib
import io
import os
import time
from pathlib import Path
from typing import List, Optional

from ..protocols import ImageExtractorProtocol
from ..rag_types import ExtractedImage
from ..utils.exceptions import ParsingError
from ..utils.logger import get_logger
from ..utils.validators import validate_pdf_path

logger = get_logger(__name__)


class FitzImageExtractor(ImageExtractorProtocol):
    """
    Image extractor using PyMuPDF (fitz).

    Args:
        parsing_config: The parsing sub-section of RAGSettings.
        cache_dir: Absolute path where extracted images are saved.
    """

    def __init__(self, parsing_config=None, cache_dir: str = "./image_cache"):
        self._config = parsing_config
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._min_width = getattr(parsing_config, "min_image_width_px", 100)
        self._min_height = getattr(parsing_config, "min_image_height_px", 100)

    # ── Public API ────────────────────────────────────────────

    @validate_pdf_path(arg_name="pdf_path")
    def extract(self, pdf_path: str) -> List[ExtractedImage]:
        """
        Extract all images from a PDF.

        Returns:
            List of ExtractedImage. Empty list if no images meet size threshold.
            Never raises on pages with no images.

        Raises:
            ParsingError: Only on file-level failure (can't open, corrupt).
        """
        t0 = time.perf_counter()
        logger.info("Extracting images from: %s", pdf_path)

        try:
            import fitz
        except ImportError as exc:
            raise ParsingError(
                "PyMuPDF (fitz) not installed. Run: pip install PyMuPDF",
                context={"hint": "pip install PyMuPDF"},
            ) from exc

        try:
            pdf_doc = fitz.open(pdf_path)
        except Exception as exc:
            raise ParsingError(
                f"fitz could not open PDF: {exc}",
                context={"pdf_path": pdf_path},
            ) from exc

        extracted: List[ExtractedImage] = []
        captions_by_page = self._collect_page_text_blocks(pdf_doc)

        for page_index in range(len(pdf_doc)):
            page = pdf_doc[page_index]
            page_number = page_index + 1
            page_images = self._extract_page_images(
                page, page_number, pdf_path, captions_by_page.get(page_number, [])
            )
            extracted.extend(page_images)

        pdf_doc.close()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Extracted %d images from %s in %.2fs",
            len(extracted), Path(pdf_path).name, elapsed,
        )
        return extracted

    # ── Private helpers ───────────────────────────────────────

    def _collect_page_text_blocks(self, pdf_doc) -> dict:
        """
        Pre-collect all text blocks per page for caption association.
        Returns dict: {page_number: [(bbox, text), ...]}
        """
        result = {}
        try:
            for page_index in range(len(pdf_doc)):
                page = pdf_doc[page_index]
                page_number = page_index + 1
                blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
                text_blocks = [
                    ((b[0], b[1], b[2], b[3]), b[4].strip())
                    for b in blocks
                    if b[6] == 0 and b[4].strip()  # type 0 = text
                ]
                result[page_number] = text_blocks
        except Exception as exc:
            logger.warning("Could not extract text blocks for caption association: %s", exc)
        return result

    def _extract_page_images(
        self,
        page,
        page_number: int,
        pdf_path: str,
        text_blocks: list,
    ) -> List[ExtractedImage]:
        """Extract and filter images from a single page."""
        images = []
        try:
            image_list = page.get_images(full=True)
        except Exception as exc:
            logger.warning("Could not get images from page %d: %s", page_number, exc)
            return images

        for img_info in image_list:
            xref = img_info[0]
            try:
                img = self._load_image(page, xref)
                if img is None:
                    continue

                image_bytes, width, height, fmt = img
                if width < self._min_width or height < self._min_height:
                    logger.debug(
                        "Skipping small image %dx%d px on page %d",
                        width, height, page_number,
                    )
                    continue

                # Get bounding box of the image on the page
                bbox = self._get_image_bbox(page, xref)

                # Associate nearest caption
                caption = self._find_nearest_caption(bbox, text_blocks)

                # Save to cache with deterministic filename
                image_path = self._save_to_cache(image_bytes, pdf_path, page_number, xref, fmt)

                images.append(ExtractedImage(
                    image_bytes=image_bytes,
                    page_number=page_number,
                    source_pdf=pdf_path,
                    caption=caption,
                    bbox=bbox,
                    width=width,
                    height=height,
                    format=fmt,
                    image_path=image_path,
                ))

            except Exception as exc:
                logger.warning(
                    "Skipping image xref=%d on page %d: %s", xref, page_number, exc
                )

        return images

    def _load_image(self, page, xref: int):
        """Load raw image bytes from PDF cross-reference."""
        try:
            import fitz
            base_image = page.parent.extract_image(xref)
            if not base_image:
                return None
            image_bytes = base_image["image"]
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            fmt = base_image.get("ext", "png").upper()
            return image_bytes, width, height, fmt
        except Exception:
            return None

    def _get_image_bbox(self, page, xref: int) -> Optional[tuple]:
        """Retrieve the bounding box of an image on its page."""
        try:
            for img in page.get_image_rects(xref):
                rect = img
                return (rect.x0, rect.y0, rect.x1, rect.y1)
        except Exception:
            pass
        return None

    def _find_nearest_caption(self, image_bbox: Optional[tuple], text_blocks: list) -> str:
        """
        Find the text block closest to the image bounding box.
        Typically image captions appear immediately below images.
        Returns empty string if no suitable caption found.
        """
        if not image_bbox or not text_blocks:
            return ""

        img_x0, img_y0, img_x1, img_y1 = image_bbox
        img_center_x = (img_x0 + img_x1) / 2

        best_caption = ""
        best_distance = float("inf")

        for block_bbox, text in text_blocks:
            bx0, by0, bx1, by1 = block_bbox
            block_center_x = (bx0 + bx1) / 2

            # Caption must be below the image
            vertical_distance = by0 - img_y1
            if vertical_distance < 0 or vertical_distance > 100:
                continue

            # Caption must be horizontally aligned with image
            horizontal_offset = abs(block_center_x - img_center_x)
            if horizontal_offset > (img_x1 - img_x0):
                continue

            distance = vertical_distance + 0.3 * horizontal_offset
            if distance < best_distance:
                best_distance = distance
                best_caption = text

        return best_caption[:300] if best_caption else ""   # cap caption length

    def _save_to_cache(
        self,
        image_bytes: bytes,
        pdf_path: str,
        page_number: int,
        xref: int,
        fmt: str,
    ) -> str:
        """Save image to cache with deterministic filename. Returns path."""
        pdf_name = Path(pdf_path).stem
        content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        extension = fmt.lower() if fmt.lower() in {"png", "jpeg", "jpg", "webp"} else "png"
        filename = f"{pdf_name}_p{page_number}_x{xref}_{content_hash}.{extension}"
        full_path = self._cache_dir / filename

        if not full_path.exists():
            with open(full_path, "wb") as f:
                f.write(image_bytes)

        return str(full_path)
