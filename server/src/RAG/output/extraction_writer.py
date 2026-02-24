"""
extraction_writer.py — Write structured extraction output after PDF parsing.

Organizes parsed content into separate folders:
  vectordb/extraction_output/<pdf_name>/
    ├── text/
    │   ├── page_1.txt
    │   ├── page_2.txt
    │   └── ...
    ├── tables/
    │   ├── page_1_table_1.md
    │   ├── page_2_table_1.md
    │   └── ...
    └── images/
        ├── <pdf>_p1_x5_abc123.png
        └── ...

USAGE (internal — called by pipeline after parsing):
    writer = ExtractionWriter(output_dir)
    report = writer.write(parsed_doc, extracted_images)
    # report = {"text_files": 5, "table_files": 3, "image_files": 8, "output_dir": "..."}
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from ..rag_types import DocumentElement, ExtractedImage, ParsedDocument
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExtractionWriter:
    """
    Writes extraction artifacts to disk in an organized structure.

    Args:
        base_output_dir: Root directory for extraction output.
                         Defaults to vectordb/extraction_output/.
    """

    def __init__(self, base_output_dir: str):
        self._base_dir = Path(base_output_dir)

    def write(
        self,
        parsed_doc: ParsedDocument,
        extracted_images: List[ExtractedImage],
    ) -> Dict[str, Any]:
        """
        Write extraction output for a single PDF.

        Returns:
            Summary dict with counts and output path.
        """
        pdf_name = Path(parsed_doc.source_pdf).stem
        doc_dir = self._base_dir / pdf_name
        text_dir = doc_dir / "text"
        table_dir = doc_dir / "tables"
        image_dir = doc_dir / "images"

        # Clean previous output for this PDF
        if doc_dir.exists():
            shutil.rmtree(doc_dir)

        text_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        text_count = self._write_text(parsed_doc.elements, text_dir)
        table_count = self._write_tables(parsed_doc.elements, table_dir)
        image_count = self._write_images(extracted_images, image_dir)

        report = {
            "text_files": text_count,
            "table_files": table_count,
            "image_files": image_count,
            "output_dir": str(doc_dir),
        }

        logger.info(
            "Extraction output for '%s': %d text, %d tables, %d images -> %s",
            pdf_name, text_count, table_count, image_count, doc_dir,
        )
        return report

    # ── Private writers ───────────────────────────────────────

    @staticmethod
    def _write_text(elements: tuple, text_dir: Path) -> int:
        """Group text elements by page and write one file per page."""
        pages: Dict[int, List[str]] = defaultdict(list)

        for elem in elements:
            if elem.element_type == "table":
                continue  # tables handled separately
            content = elem.content.strip()
            if not content:
                continue

            label = elem.element_type.upper()
            if elem.element_type == "heading":
                prefix = "#" * max(elem.level, 1) + " "
                pages[elem.page_number].append(f"{prefix}{content}")
            elif elem.element_type == "list":
                pages[elem.page_number].append(f"[LIST]\n{content}")
            elif elem.element_type == "image_caption":
                pages[elem.page_number].append(f"[CAPTION] {content}")
            elif elem.element_type == "pricing_block":
                pages[elem.page_number].append(f"[PRICING]\n{content}")
            else:
                pages[elem.page_number].append(content)

        count = 0
        for page_num in sorted(pages.keys()):
            filepath = text_dir / f"page_{page_num}.txt"
            filepath.write_text(
                f"{'='*60}\n"
                f"  PAGE {page_num}\n"
                f"{'='*60}\n\n"
                + "\n\n".join(pages[page_num]) + "\n",
                encoding="utf-8",
            )
            count += 1
        return count

    @staticmethod
    def _write_tables(elements: tuple, table_dir: Path) -> int:
        """Write each table as a Markdown file."""
        count = 0
        table_counter: Dict[int, int] = defaultdict(int)

        for elem in elements:
            if elem.element_type != "table":
                continue

            page = elem.page_number
            table_counter[page] += 1
            idx = table_counter[page]

            filepath = table_dir / f"page_{page}_table_{idx}.md"

            lines = [f"# Table {idx} — Page {page}\n"]

            if elem.table_data:
                # Render as Markdown table
                for row_i, row in enumerate(elem.table_data):
                    cells = [c.replace("|", "\\|") for c in row]
                    lines.append("| " + " | ".join(cells) + " |")
                    if row_i == 0:
                        lines.append("| " + " | ".join("---" for _ in cells) + " |")
            elif elem.content.strip():
                # Fallback: raw text
                lines.append(elem.content.strip())
            else:
                lines.append("*(empty table)*")

            filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
            count += 1

        return count

    @staticmethod
    def _write_images(
        extracted_images: List[ExtractedImage], image_dir: Path
    ) -> int:
        """Copy extracted images into the output directory."""
        count = 0
        for img in extracted_images:
            src = Path(img.image_path) if img.image_path else None
            if src and src.exists():
                dst = image_dir / src.name
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
                count += 1
            elif img.image_bytes:
                ext = img.format.lower() if img.format else "png"
                fname = f"page_{img.page_number}_{img.image_id[:8]}.{ext}"
                dst = image_dir / fname
                dst.write_bytes(img.image_bytes)
                count += 1
        return count

    def print_summary(
        self,
        parsed_doc: ParsedDocument,
        extracted_images: List[ExtractedImage],
        report: Dict[str, Any],
    ) -> None:
        """Print a formatted extraction summary to stdout."""
        pdf_name = Path(parsed_doc.source_pdf).name

        # Count elements by type
        type_counts: Dict[str, int] = defaultdict(int)
        for elem in parsed_doc.elements:
            type_counts[elem.element_type] += 1

        print(f"\n{'='*60}")
        print(f"  EXTRACTION OUTPUT -- {pdf_name}")
        print(f"{'='*60}")
        print(f"  Total pages:  {parsed_doc.total_pages}")
        print(f"  Output dir:   {report['output_dir']}")
        print()

        print("  -- Text Elements ---------------------")
        for etype in ["heading", "paragraph", "list", "image_caption", "pricing_block", "map"]:
            if type_counts.get(etype, 0) > 0:
                print(f"    {etype:<18} {type_counts[etype]:>4}")
        print(f"    {'TOTAL':<18} {sum(v for k,v in type_counts.items() if k != 'table'):>4}")
        print(f"    Files written:   {report['text_files']}")

        print()
        print("  -- Tables ----------------------------")
        print(f"    Tables found:    {type_counts.get('table', 0):>4}")
        print(f"    Files written:   {report['table_files']}")

        print()
        print("  -- Images ----------------------------")
        print(f"    Images extracted:{len(extracted_images):>4}")
        print(f"    Files written:   {report['image_files']}")

        if parsed_doc.parse_warnings:
            print()
            print(f"  -- Warnings ({len(parsed_doc.parse_warnings)}) ---")
            for w in parsed_doc.parse_warnings[:5]:
                print(f"    ! {w}")

        print(f"\n{'='*60}\n")
