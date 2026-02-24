"""
verify_rag.py — Quick smoke test for the RAG pipeline components.

Tests:
  1. Config loads correctly
  2. All imports resolve
  3. ModelManager loads BGE + CLIP from local paths
  4. Text embedder produces 768-dim vectors
  5. Image embedder produces 512-dim vectors
  6. Pipeline instantiates without error
"""

import sys
import time

def main():
    t0 = time.perf_counter()
    errors = []

    # 1. Config
    print("[1/6] Loading config...")
    try:
        from server.src.RAG.config.settings import get_settings, reset_settings
        reset_settings()
        cfg = get_settings()
        print(f"  OK — device={cfg.effective_device}, models dir={cfg.models.text_embedding.local_path}")
    except Exception as e:
        errors.append(f"Config: {e}")
        print(f"  FAIL: {e}")

    # 2. All imports
    print("[2/6] Checking imports...")
    try:
        from server.src.RAG import RAGPipeline, RAGResult, RAGContext
        from server.src.RAG.rag_types import (
            DocumentElement, ParsedDocument, ExtractedImage,
            ChunkMetadata, StructuredChunk, ImageNodeData,
            ProcessedQuery, RetrievalCandidate, RAGContext, RAGResult,
        )
        from server.src.RAG.protocols import (
            ParserProtocol, ImageExtractorProtocol, ChunkerProtocol,
            TextEmbedderProtocol, ImageEmbedderProtocol,
            IndexProtocol, RankerProtocol, AssemblerProtocol,
            QueryProcessorProtocol, RetrieverProtocol, FilterProtocol,
        )
        print("  OK — all imports resolved.")
    except Exception as e:
        errors.append(f"Imports: {e}")
        print(f"  FAIL: {e}")

    # 3. ModelManager
    print("[3/6] Loading models via ModelManager...")
    mm = None  # initialise before try so steps 4-5 can guard safely
    try:
        from server.src.RAG.embedding.model_manager import ModelManager
        mm = ModelManager(models_config=cfg.models, device=cfg.effective_device)

        bge = mm.get_text_model()
        print(f"  BGE loaded: {type(bge).__name__}")

        clip_model, clip_proc = mm.get_clip_model()
        print(f"  CLIP loaded: {type(clip_model).__name__}")
    except Exception as e:
        errors.append(f"ModelManager: {e}")
        print(f"  FAIL: {e}")

    # 4. Text embedder
    print("[4/6] Testing text embedder (BGE 768-dim)...")
    if mm is None:
        errors.append("TextEmbedder: skipped (ModelManager failed)")
        print("  SKIP — ModelManager unavailable")
    else:
        try:
            from server.src.RAG.embedding.text_embedder import BGETextEmbedder
            te = BGETextEmbedder(mm, cfg.models.text_embedding)
            vec = te.embed_single("sunset over lake palace in udaipur")
            assert vec.shape == (768,), f"Expected (768,), got {vec.shape}"
            print(f"  OK — shape={vec.shape}, norm={sum(vec**2)**0.5:.4f}")
        except Exception as e:
            errors.append(f"TextEmbedder: {e}")
            print(f"  FAIL: {e}")

    # 5. Image embedder (text mode)
    print("[5/6] Testing image embedder (CLIP 512-dim)...")
    if mm is None:
        errors.append("ImageEmbedder: skipped (ModelManager failed)")
        print("  SKIP — ModelManager unavailable")
    else:
        try:
            from server.src.RAG.embedding.image_embedder import CLIPImageEmbedder
            ie = CLIPImageEmbedder(mm, cfg.models.image_embedding)
            vec = ie.embed_text("a beautiful mountain view at sunrise")
            assert vec.shape == (512,), f"Expected (512,), got {vec.shape}"
            print(f"  OK — shape={vec.shape}, norm={sum(vec**2)**0.5:.4f}")
        except Exception as e:
            errors.append(f"ImageEmbedder: {e}")
            print(f"  FAIL: {e}")

    # 6. Pipeline instantiation
    print("[6/6] Instantiating RAGPipeline...")
    try:
        from server.src.RAG.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        health = pipeline.health_check()
        print(f"  OK — health: {health}")
    except Exception as e:
        errors.append(f"Pipeline: {e}")
        print(f"  FAIL: {e}")

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    if not errors:
        print(f"  ALL TESTS PASSED in {elapsed:.1f}s")
    else:
        print(f"  {len(errors)} FAILURE(S):")
        for err in errors:
            print(f"    - {err}")
    print(f"{'='*60}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
