"""
routes/upload.py — File upload endpoint for images and PDFs.

POST /api/upload  → upload a file (image or PDF), return analysis
"""

from __future__ import annotations

import base64
import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["upload"])

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
ALLOWED_PDF_TYPES = {"application/pdf"}
ALLOWED_TYPES = ALLOWED_IMAGE_TYPES | ALLOWED_PDF_TYPES


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    file_url: str
    message: str


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(""),
    message: str = Form(""),
):
    """
    Upload an image or PDF file.
    - Images are saved and can be sent to the image_analysis_agent.
    - PDFs are saved for context extraction.
    Returns a file_id and URL for the frontend to reference.
    """
    if file.content_type not in ALLOWED_TYPES:
        return UploadResponse(
            file_id="",
            filename=file.filename or "",
            file_type=file.content_type or "",
            file_url="",
            message=f"Unsupported file type: {file.content_type}. Allowed: images (png, jpg, webp, gif) and PDFs.",
        )

    # Generate unique filename
    ext = Path(file.filename or "file").suffix
    file_id = str(uuid.uuid4())
    saved_name = f"{file_id}{ext}"
    save_path = UPLOAD_DIR / saved_name

    # Save file
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    file_type = "image" if file.content_type in ALLOWED_IMAGE_TYPES else "pdf"
    file_url = f"/api/uploads/{saved_name}"

    logger.info("Uploaded %s: %s (%s bytes)", file_type, saved_name, len(content))

    return UploadResponse(
        file_id=file_id,
        filename=file.filename or saved_name,
        file_type=file_type,
        file_url=file_url,
        message=f"{file_type.capitalize()} uploaded successfully.",
    )


@router.get("/uploads/{filename}")
async def serve_upload(filename: str):
    """Serve an uploaded file."""
    from fastapi.responses import FileResponse

    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
