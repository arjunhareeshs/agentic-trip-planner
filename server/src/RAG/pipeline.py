"""
pipeline.py — The RAG Pipeline error boundary and sole public entry point.

════════════════════════════════════════════════════════════════════════════

PUBLIC API (everything external code should use):

    from RAG import RAGPipeline, RAGResult, RAGContext

    pipeline = RAGPipeline()

    # Ingest a PDF brochure:
    result = pipeline.ingest("/path/to/brochure.pdf")
    if result.success:
        print(f"Ingested {result.data['chunks']} chunks, {result.data['images']} images")
    else:
        print(f"Ingest failed: {result.error}")

    # Query the pipeline:
    result = pipeline.query("Show me lake views in Udaipur at sunset")
    if result.success:
        context: RAGContext = result.data
        print(context.assembled_prompt)
        print(context.image_paths)
    else:
        print(f"Query failed: {result.error}")

════════════════════════════════════════════════════════════════════════════

ERROR BOUNDARY:
  • ALL exceptions are caught here. None propagate outside this class.
  • Library errors, internal module errors, and network errors are all
    caught and returned as RAGResult(success=False, error=<message>).
  • External code never needs try/except around pipeline calls.

WIRING:
  All modules are instantiated here and injected into each other.
  To swap any component: change default.yaml, not this file.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from .assembly.context_assembler import ContextAssembler
from .chunking.structure_chunker import StructureChunker
from .config.settings import get_settings
from .embedding.image_embedder import CLIPImageEmbedder
from .embedding.model_manager import ModelManager
from .embedding.text_embedder import BGETextEmbedder
from .indexing.caption_index import CaptionVectorIndex
from .indexing.image_index import ImageVectorIndex
from .indexing.text_index import TextVectorIndex
from .output.extraction_writer import ExtractionWriter
from .parsing.docling_parser import DoclingParser
from .parsing.image_extractor import FitzImageExtractor
from .parsing.metadata_extractor import MetadataExtractor
from .query.query_processor import QueryProcessor
from .ranking.cross_encoder_reranker import CrossEncoderReranker
from .rag_types import (
    ChunkMetadata,
    ImageNodeData,
    ProcessedQuery,
    RAGContext,
    RAGResult,
    StructuredChunk,
)
from .utils.exceptions import RAGError
from .utils.logger import get_logger, initialize_logging


class RAGPipeline:
    """
    Dual-modal RAG pipeline: text + vision.
    Error boundary — exceptions never propagate past this class.
    """

    def __init__(self):
        """Initialize the pipeline. Wire all components from default.yaml config."""
        self._settings = get_settings()
        initialize_logging(self._settings.log_level)
        self._logger = get_logger(__name__)
        self._logger.info("Initializing RAGPipeline...")

        # Resolve paths from config
        base_dir = Path(__file__).parent
        persist_dir = str(base_dir / self._settings.indexing.persist_dir)
        image_cache_dir = str(base_dir / self._settings.indexing.image_cache_dir)

        # ── Component wiring ──────────────────────────────────────

        # Device
        device = self._settings.effective_device
        self._logger.info("Using device: %s", device)

        # Model Manager (shared across all embedding components)
        self._model_manager = ModelManager(
            models_config=self._settings.models,
            device=device,
            base_dir=base_dir,
        )

        # Embedders
        self._text_embedder = BGETextEmbedder(
            self._model_manager, self._settings.models.text_embedding
        )
        self._image_embedder = CLIPImageEmbedder(
            self._model_manager, self._settings.models.image_embedding
        )

        # Parsers
        self._parser = DoclingParser(self._settings.parsing)
        self._image_extractor = FitzImageExtractor(
            self._settings.parsing, cache_dir=image_cache_dir
        )
        self._metadata_extractor = MetadataExtractor(self._settings.metadata)

        # Chunker
        self._chunker = StructureChunker(self._settings.chunking)

        # Indices
        self._text_index = TextVectorIndex(
            self._text_embedder, self._settings.indexing, persist_dir
        )
        self._image_index = ImageVectorIndex(
            self._image_embedder, self._settings.indexing, persist_dir
        )
        self._caption_index = CaptionVectorIndex(
            self._text_embedder, self._settings.indexing, persist_dir
        )

        # Query processor
        self._query_processor = QueryProcessor(
            self._text_index,
            self._image_index,
            self._caption_index,
            self._text_embedder,
            self._image_embedder,
            self._metadata_extractor,
            self._settings.retrieval,
        )

        # Ranker
        self._reranker = CrossEncoderReranker(self._model_manager)

        # Assembler
        self._assembler = ContextAssembler(self._settings.retrieval)

        # Extraction output writer (dir set externally via set_extraction_dir)
        self._extraction_writer = None

        # Keep track of all ingested chunks (for supporting text lookup during assembly)
        self._all_text_chunks: List[StructuredChunk] = []

        self._logger.info("RAGPipeline initialized successfully.")

    def set_extraction_dir(self, extraction_dir: str) -> None:
        """Set the directory where extraction output (text/tables/images) is written."""
        self._extraction_writer = ExtractionWriter(extraction_dir)
        self._logger.info("Extraction output dir: %s", extraction_dir)

    # ══════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════

    def ingest(self, pdf_paths: List[str] | str) -> RAGResult:
        """
        Ingest one or more PDF brochures into the pipeline.

        Args:
            pdf_paths: Absolute path or list of absolute paths to .pdf files.

        Returns:
            RAGResult(
                success=True,
                data={
                    "total": N,
                    "success": M,
                    "failed": [...],
                    "chunks": X,
                    "images": Y
                }
            )
        """
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]

        t0 = time.perf_counter()
        self._logger.info("=== BATCH INGEST START: %d files ===", len(pdf_paths))

        total_chunks = 0
        total_images = 0
        successful_files = 0
        failed_files = []
        extraction_reports = []

        for path in pdf_paths:
            try:
                result = self._ingest_single(path)
                successful_files += 1
                total_chunks += result["chunks"]
                total_images += result["images"]
                extraction_reports.append(result["extraction"])
            except Exception as exc:
                self._logger.error("Failed to ingest %s: %s", path, exc)
                failed_files.append({"path": path, "error": str(exc)})

        elapsed = time.perf_counter() - t0
        self._logger.info(
            "=== BATCH INGEST COMPLETE: %d/%d success in %.1fs ===",
            successful_files, len(pdf_paths), elapsed
        )

        return RAGResult(
            success=True if successful_files > 0 else False,
            data={
                "total": len(pdf_paths),
                "success": successful_files,
                "failed": failed_files,
                "chunks": total_chunks,
                "images": total_images,
                "elapsed_sec": round(elapsed, 2),
                "extraction": extraction_reports,
            },
            error=f"All {len(pdf_paths)} files failed." if successful_files == 0 else None
        )

    def _ingest_single(self, pdf_path: str) -> Dict[str, Any]:
        """Internal helper for single-file ingestion."""
        # 1. Parse
        parsed_doc = self._parser.parse(pdf_path)

        # 2. Extract images
        extracted_images = []
        try:
            extracted_images = self._image_extractor.extract(pdf_path)
        except Exception as exc:
            self._logger.warning("Image extraction failed for %s: %s", pdf_path, exc)

        # 3. Write extraction output (text / tables / images separated)
        extraction_report = {}
        if self._extraction_writer is not None:
            extraction_report = self._extraction_writer.write(parsed_doc, extracted_images)
            self._extraction_writer.print_summary(
                parsed_doc, extracted_images, extraction_report
            )

        # 4. Chunk text
        chunks = self._chunker.chunk(parsed_doc)

        # 5. Enrich chunk metadata
        chunks = self._enrich_chunk_metadata(chunks)
        self._all_text_chunks.extend(chunks)

        # 6. Build image nodes
        image_nodes = self._build_image_nodes(extracted_images)

        # 7. Index
        self._text_index.add(chunks)
        if image_nodes:
            self._image_index.add(image_nodes)
            self._caption_index.add(image_nodes)

        return {
            "chunks": len(chunks),
            "images": len(extracted_images),
            "extraction": extraction_report,
        }

    def query(self, raw_query: str) -> RAGResult:
        """
        Query the pipeline with a natural language question.

        Steps:
          1. Validate + detect metadata from query
          2. Parallel retrieval across text, image, caption indices
          3. Cross-encoder reranking
          4. Context assembly

        Args:
            raw_query: User query string (1-2000 chars).

        Returns:
            RAGResult(
                success=True,
                data=RAGContext(
                    assembled_prompt, image_paths, retrieved_text_nodes, ...
                )
            )
            or
            RAGResult(
                success=False,
                error="<description>"
            )
        """
        t0 = time.perf_counter()
        self._logger.info("=== QUERY START: %s ===", raw_query[:80])

        try:
            # 1. Validate and enrich query
            raw_query = raw_query.strip()
            if not raw_query:
                return RAGResult(
                    success=False,
                    error="Query must not be empty.",
                    error_code="QUERY_EMPTY",
                )

            detected_meta = self._metadata_extractor.extract_from_query(raw_query)
            processed_query = ProcessedQuery(
                text=raw_query,
                detected_metadata=detected_meta,
            )

            # 2. Retrieve
            t_retrieve = time.perf_counter()
            candidates = self._query_processor.process(processed_query)
            self._log_timing("retrieve", t_retrieve)

            if not candidates:
                self._logger.warning("No candidates found after retrieval.")
                return RAGResult(
                    success=True,
                    data=RAGContext(
                        query=raw_query,
                        retrieved_text_nodes=(),
                        retrieved_image_nodes=(),
                        assembled_prompt="No relevant content found for this query.",
                        token_count=0,
                        image_paths=(),
                        source_pdfs=(),
                    ),
                )

            # 3. Rerank
            t_rerank = time.perf_counter()
            reranked = self._reranker.rerank(
                raw_query,
                candidates,
                top_k=self._settings.retrieval.rerank_top_k,
            )
            self._log_timing("rerank", t_rerank)

            # 4. Assemble context
            t_assemble = time.perf_counter()
            context = self._assembler.assemble(
                raw_query, reranked, self._all_text_chunks if self._all_text_chunks else []
            )
            self._log_timing("assemble", t_assemble)

            elapsed = time.perf_counter() - t0
            self._logger.info(
                "=== QUERY COMPLETE: %d text + %d image nodes, ~%d tokens in %.1fs ===",
                len(context.retrieved_text_nodes),
                len(context.retrieved_image_nodes),
                context.token_count,
                elapsed,
            )

            return RAGResult(success=True, data=context)

        except RAGError as exc:
            self._logger.error("Query RAGError [%s]: %s", exc.error_code, exc)
            return RAGResult(success=False, error=str(exc), error_code=exc.error_code)

        except Exception as exc:
            self._logger.exception("Query unexpected error:")
            return RAGResult(
                success=False,
                error=f"Unexpected query error: {type(exc).__name__}: {exc}",
                error_code="UNEXPECTED_ERROR",
            )

    def health_check(self) -> Dict[str, Any]:
        """
        Run model health checks. Returns status dict.
        Safe to call at any time — does not raise.
        """
        try:
            return self._model_manager.health_check()
        except Exception as exc:
            return {"error": str(exc)}

    # ══════════════════════════════════════════════════════════
    #  Private helpers
    # ══════════════════════════════════════════════════════════

    def _enrich_chunk_metadata(
        self, chunks: List[StructuredChunk]
    ) -> List[StructuredChunk]:
        """
        Add metadata (city, scene_type, emotion_tags, etc.) to each chunk
        by running MetadataExtractor on chunk content.
        Returns a new list of enriched StructuredChunk objects.
        """
        enriched = []
        for chunk in chunks:
            detected = self._metadata_extractor.extract(chunk.content)
            old_meta = chunk.metadata
            new_meta = ChunkMetadata(
                source_pdf=old_meta.source_pdf,
                page_number=old_meta.page_number,
                section_hierarchy=old_meta.section_hierarchy,
                element_type=old_meta.element_type,
                city=detected.get("city", ""),
                scene_type=detected.get("scene_type", ""),
                crowd_level=detected.get("crowd_level", ""),
                lighting=detected.get("lighting", ""),
                emotion_tags=tuple(detected.get("emotion_tags", [])),
            )
            enriched.append(StructuredChunk(
                content=chunk.content,
                metadata=new_meta,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
            ))
        return enriched

    def _build_image_nodes(
        self, extracted_images: list
    ) -> List[ImageNodeData]:
        """
        Convert ExtractedImage objects to ImageNodeData objects,
        embedding each image with CLIP and extracting metadata from caption.
        Failures are skipped with a warning, not raised.
        """
        nodes = []
        for img in extracted_images:
            try:
                image_vector = self._image_embedder.embed_image(img.image_bytes)
                caption_meta = self._metadata_extractor.extract(img.caption)

                nodes.append(ImageNodeData(
                    image_id=img.image_id,
                    image_path=getattr(img, "image_path", "") or "",
                    caption=img.caption,
                    image_vector=image_vector,
                    city=caption_meta.get("city", ""),
                    scene_type=caption_meta.get("scene_type", ""),
                    crowd_level=caption_meta.get("crowd_level", ""),
                    lighting=caption_meta.get("lighting", ""),
                    emotion_tags=tuple(caption_meta.get("emotion_tags", [])),
                    source_pdf=img.source_pdf,
                    page_number=img.page_number,
                ))
            except Exception as exc:
                self._logger.warning(
                    "Skipping image on page %d (embedding failed): %s",
                    img.page_number, exc,
                )
        return nodes

    def _log_timing(self, stage: str, t_start: float) -> None:
        """Log stage timing if log_timing is enabled in config."""
        if self._settings.log_timing:
            elapsed = time.perf_counter() - t_start
            self._logger.debug("Timing [%s]: %.3fs", stage, elapsed)
