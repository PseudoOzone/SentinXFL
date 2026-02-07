"""
SentinXFL - File Upload API Routes
=====================================

Handles CSV/data file uploads from client banks.
Files are validated, processed, and fed into the ML pipeline.

Author: Anshuman Bakshi
"""

import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
from pydantic import BaseModel

from sentinxfl.core.logging import get_logger
from sentinxfl.api.routes.auth import require_auth

log = get_logger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

# ============================================
# In-memory upload registry
# ============================================
_uploads: dict[str, dict] = {}


class UploadInfo(BaseModel):
    upload_id: str
    filename: str
    bank_id: str
    file_size: int
    status: str  # "uploaded" | "processing" | "completed" | "error"
    uploaded_at: str
    row_count: Optional[int] = None
    fraud_count: Optional[int] = None
    error_message: Optional[str] = None


# ============================================
# Upload Endpoints
# ============================================


@router.post("/upload", tags=["upload"])
async def upload_file(
    file: UploadFile = File(...),
    bank_id: str = Form(...),
    description: str = Form(default=""),
    user: dict = Depends(require_auth),
):
    """
    Upload a CSV/data file from a client bank.
    Validates file type, size, and saves for processing.
    """
    # Validate file type
    if not file.filename or not file.filename.endswith((".csv", ".parquet")):
        raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported")

    # Validate user access
    if user["role"] == "client" and user.get("bank_id") != bank_id:
        raise HTTPException(status_code=403, detail="Cannot upload for a different bank")

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large (max {MAX_FILE_SIZE // (1024*1024)}MB)")

    upload_id = f"upload-{uuid.uuid4().hex[:12]}"
    bank_dir = UPLOAD_DIR / bank_id
    bank_dir.mkdir(parents=True, exist_ok=True)

    save_path = bank_dir / f"{upload_id}_{file.filename}"
    save_path.write_bytes(content)

    upload_info = {
        "upload_id": upload_id,
        "filename": file.filename,
        "bank_id": bank_id,
        "file_size": len(content),
        "status": "uploaded",
        "uploaded_at": datetime.utcnow().isoformat(),
        "file_path": str(save_path),
        "description": description,
        "uploaded_by": user["user_id"],
        "row_count": None,
        "fraud_count": None,
        "error_message": None,
    }
    _uploads[upload_id] = upload_info

    log.info("File uploaded: %s (%d bytes) for bank %s", file.filename, len(content), bank_id)

    return UploadInfo(**{k: v for k, v in upload_info.items() if k in UploadInfo.model_fields})


@router.get("/upload", tags=["upload"])
async def list_uploads(
    bank_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    user: dict = Depends(require_auth),
):
    """List uploads with optional filters."""
    results = list(_uploads.values())

    # Client can only see their bank's uploads
    if user["role"] == "client":
        results = [u for u in results if u["bank_id"] == user.get("bank_id")]
    elif bank_id:
        results = [u for u in results if u["bank_id"] == bank_id]

    if status:
        results = [u for u in results if u["status"] == status]

    results.sort(key=lambda u: u["uploaded_at"], reverse=True)
    results = results[:limit]

    return {
        "uploads": [
            UploadInfo(**{k: v for k, v in u.items() if k in UploadInfo.model_fields}).model_dump()
            for u in results
        ],
        "count": len(results),
    }


@router.get("/upload/{upload_id}", tags=["upload"])
async def get_upload(upload_id: str, user: dict = Depends(require_auth)):
    """Get upload details."""
    upload = _uploads.get(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Access check
    if user["role"] == "client" and upload["bank_id"] != user.get("bank_id"):
        raise HTTPException(status_code=403, detail="Access denied")

    return UploadInfo(**{k: v for k, v in upload.items() if k in UploadInfo.model_fields})


@router.post("/upload/{upload_id}/process", tags=["upload"])
async def process_upload(upload_id: str, user: dict = Depends(require_auth)):
    """
    Trigger processing of an uploaded file.
    Validates the CSV, counts rows/fraud, prepares for ML pipeline.
    """
    upload = _uploads.get(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    if upload["status"] != "uploaded":
        raise HTTPException(status_code=400, detail=f"Upload status is '{upload['status']}', expected 'uploaded'")

    upload["status"] = "processing"

    try:
        file_path = Path(upload["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Process based on file type
        if file_path.suffix == ".csv":
            import polars as pl

            df = pl.read_csv(file_path)
            row_count = len(df)

            # Try to detect fraud column
            fraud_col = None
            for col_name in ["is_fraud", "fraud", "Class", "isFraud", "label", "target"]:
                if col_name in df.columns:
                    fraud_col = col_name
                    break

            fraud_count = 0
            if fraud_col:
                fraud_count = int(df[fraud_col].sum())

            upload["row_count"] = row_count
            upload["fraud_count"] = fraud_count
            upload["status"] = "completed"

            log.info("Processed upload %s: %d rows, %d fraud", upload_id, row_count, fraud_count)
        else:
            upload["status"] = "completed"
            upload["row_count"] = 0

    except Exception as e:
        upload["status"] = "error"
        upload["error_message"] = str(e)
        log.error("Error processing upload %s: %s", upload_id, e)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return UploadInfo(**{k: v for k, v in upload.items() if k in UploadInfo.model_fields})


@router.delete("/upload/{upload_id}", tags=["upload"])
async def delete_upload(upload_id: str, user: dict = Depends(require_auth)):
    """Delete an upload."""
    upload = _uploads.get(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    # Access check
    if user["role"] == "client" and upload["bank_id"] != user.get("bank_id"):
        raise HTTPException(status_code=403, detail="Access denied")

    # Delete file
    try:
        file_path = Path(upload["file_path"])
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        log.warning("Could not delete file for upload %s: %s", upload_id, e)

    del _uploads[upload_id]
    return {"message": f"Upload {upload_id} deleted"}
