"""
FastAPI REST API backend for the Image Caption Generator.

Endpoints:
  GET  /                  — Health check
  POST /generate-caption  — Upload image, return generated caption
  GET  /history           — Retrieve caption generation history
"""

import logging
import sys
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# Ensure backend root is on sys.path when running as script
BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import (
    ALLOWED_EXTENSIONS,
    API_HOST,
    API_PORT,
    LOGS_DIR,
    MAX_UPLOAD_SIZE_MB,
    UPLOADS_DIR,
    get_model_path,
)
from database import get_database
from predict import CaptionPredictor, get_predictor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Image Caption Generator API",
    description="Generate natural language captions for images using VGG16 + LSTM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str


class CaptionResponse(BaseModel):
    caption: str
    confidence: Optional[float] = None
    image_name: Optional[str] = None


class HistoryRecord(BaseModel):
    id: int
    image_name: str
    caption: str
    confidence: Optional[float]
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_image(file: UploadFile) -> None:
    """Raise HTTPException if uploaded file is not a valid image type."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )


def _load_pil_image(file_bytes: bytes) -> Image.Image:
    """Decode bytes to a PIL RGB image."""
    try:
        return Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns service status and whether the trained model is available.
    """
    model_loaded = get_model_path().exists()
    return HealthResponse(
        status="running",
        model_loaded=model_loaded,
        message=(
            "Image Caption Generator API is running."
            if model_loaded
            else "API running but model not trained yet. Run: python train.py"
        ),
    )


@app.post("/generate-caption", response_model=CaptionResponse, tags=["Caption"])
async def generate_caption(
    file: UploadFile = File(..., description="Image file (JPG, PNG, JPEG)"),
) -> CaptionResponse:
    """
    Upload an image and receive an AI-generated caption.

    - Accepts multipart/form-data with an image file
    - Extracts VGG16 features
    - Generates caption word-by-word via LSTM
    - Stores result in SQLite database

    **Example response:**
    ```json
    {"caption": "A dog is playing with a ball in the grass.", "confidence": 0.72}
    ```
    """
    _validate_image(file)

    # Read and size-check file
    contents = await file.read()
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB} MB",
        )

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Decode image
    pil_image = _load_pil_image(contents)
    original_name = file.filename or "upload.jpg"
    saved_name = f"{uuid.uuid4().hex}_{original_name}"

    # Save upload for audit trail
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADS_DIR / saved_name
    try:
        pil_image.save(save_path)
    except Exception as exc:
        logger.warning("Could not save upload: %s", exc)

    # Generate caption
    try:
        predictor = get_predictor()
        result = predictor.predict_from_pil(pil_image)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train the model first: python train.py",
        )
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Caption generation failed: {exc}")

    caption = result["caption"]
    confidence = result.get("confidence")

    # Persist to database
    try:
        get_database().save(
            image_name=saved_name,
            caption=caption,
            confidence=confidence,
        )
    except Exception as exc:
        logger.warning("Database save failed: %s", exc)

    logger.info("Caption generated for '%s': %s", original_name, caption)

    return CaptionResponse(
        caption=caption,
        confidence=confidence,
        image_name=original_name,
    )


@app.get("/history", response_model=list[HistoryRecord], tags=["History"])
async def get_history(limit: int = 20) -> list[HistoryRecord]:
    """
    Retrieve recent caption generation history from the database.

    Args:
        limit: Maximum records to return (default 20, max 100).
    """
    limit = min(max(1, limit), 100)
    records = get_database().get_all(limit=limit)
    return [HistoryRecord(**r) for r in records]


@app.delete("/history/{record_id}", tags=["History"])
async def delete_history_record(record_id: int) -> dict:
    """Delete a single history record by ID."""
    deleted = get_database().delete(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Record not found.")
    return {"status": "deleted", "id": record_id}


@app.delete("/history", tags=["History"])
async def clear_history() -> dict:
    """Clear all caption history records."""
    count = get_database().clear_all()
    return {"status": "cleared", "deleted": count}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch unhandled exceptions and return a JSON error response."""
    logger.exception("Unhandled error on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
