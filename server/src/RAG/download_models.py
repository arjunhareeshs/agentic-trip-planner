"""
download_models.py — One-time script to download and cache embedding models locally.

Downloads:
  1. BAAI/bge-base-en-v1.5  (text embedding, 768-dim)
  2. openai/clip-vit-base-patch32  (vision+text embedding, 512-dim)

Models are saved into ./models/ relative to this script.
Run once: python download_models.py
"""

import os
import sys
import time
from pathlib import Path

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_LOCAL_DIR = MODELS_DIR / "bge-base-en-v1.5"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_LOCAL_DIR = MODELS_DIR / "clip-vit-b-32"


def download_bge():
    """Download BGE text embedding model."""
    print(f"\n{'='*60}")
    print(f"  Downloading: {BGE_MODEL_NAME}")
    print(f"  Save to:     {BGE_LOCAL_DIR}")
    print(f"{'='*60}")

    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(BGE_MODEL_NAME)
    model.save(str(BGE_LOCAL_DIR))

    # Quick sanity check
    test = model.encode(["hello world"], normalize_embeddings=True)
    assert test.shape == (1, 768), f"Unexpected shape: {test.shape}"

    elapsed = time.perf_counter() - t0
    print(f"  BGE model downloaded and verified in {elapsed:.1f}s")
    print(f"  Output dim: 768, test passed.")
    return True


def download_clip():
    """Download CLIP vision+text embedding model."""
    print(f"\n{'='*60}")
    print(f"  Downloading: {CLIP_MODEL_NAME}")
    print(f"  Save to:     {CLIP_LOCAL_DIR}")
    print(f"{'='*60}")

    from transformers import CLIPModel, CLIPProcessor
    import torch

    t0 = time.perf_counter()

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    # Save locally
    CLIP_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(CLIP_LOCAL_DIR))
    processor.save_pretrained(str(CLIP_LOCAL_DIR))

    # Quick sanity check with text
    inputs = processor(text=["hello world"], return_tensors="pt", padding=True)  # type: ignore[operator]
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    assert features.shape[-1] == 512, f"Unexpected CLIP dim: {features.shape}"

    elapsed = time.perf_counter() - t0
    print(f"  CLIP model downloaded and verified in {elapsed:.1f}s")
    print(f"  Output dim: 512, test passed.")
    return True


def main():
    print("Model Download Script for RAG Pipeline")
    print(f"Models directory: {MODELS_DIR}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    success = True

    try:
        download_bge()
    except Exception as e:
        print(f"  ERROR downloading BGE: {e}")
        success = False

    try:
        download_clip()
    except Exception as e:
        print(f"  ERROR downloading CLIP: {e}")
        success = False

    if success:
        print(f"\n{'='*60}")
        print("  All models downloaded successfully!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("  Some models failed to download. Check errors above.")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
