"""
Service layer for the Streamlit frontend.

Handles caption generation (direct backend or API) and database access.
"""

import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import requests
from PIL import Image

# Add backend to path for direct imports
BACKEND_ROOT = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB, get_model_path
from database import get_database
from utils import format_timestamp, get_model_info


def is_model_available() -> bool:
    """Check if trained model and tokenizer exist."""
    from config import TOKENIZER_PATH
    return get_model_path().exists() and TOKENIZER_PATH.exists()


def generate_caption_direct(pil_image: Image.Image) -> Dict:
    """
    Generate caption using backend modules directly (no API required).

    Returns:
        Dict with caption, confidence keys.
    """
    from predict import get_predictor
    return get_predictor().predict_from_pil(pil_image)


def generate_caption_api(
    image_bytes: bytes,
    filename: str,
    api_url: str,
) -> Dict:
    """POST image to FastAPI /generate-caption endpoint."""
    files = {"file": (filename, image_bytes, "image/jpeg")}
    response = requests.post(f"{api_url}/generate-caption", files=files, timeout=120)
    data = response.json()
    if not response.ok:
        raise RuntimeError(data.get("detail", f"API error {response.status_code}"))
    return data


def validate_upload(filename: str, file_size: int) -> Optional[str]:
    """
    Validate uploaded file. Returns error message or None if valid.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return f"Invalid file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        return f"File too large. Maximum size: {MAX_UPLOAD_SIZE_MB} MB"
    return None


def save_prediction(image_name: str, caption: str, confidence: Optional[float] = None) -> int:
    """Save prediction to SQLite database."""
    return get_database().save(image_name, caption, confidence)


def get_history(limit: int = 50) -> List[Dict]:
    """Fetch prediction history from database."""
    records = get_database().get_all(limit=limit)
    for record in records:
        date_str, time_str = format_timestamp(record.get("created_at", ""))
        record["date"] = date_str
        record["time"] = time_str
    return records


def delete_history_record(record_id: int) -> bool:
    """Delete a single history record."""
    return get_database().delete(record_id)


def clear_all_history() -> int:
    """Clear all history records."""
    return get_database().clear_all()


def get_api_health(api_url: str) -> Dict:
    """Check FastAPI backend health."""
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        return response.json() if response.ok else {"status": "error"}
    except requests.RequestException:
        return {"status": "offline"}


def image_to_bytes(pil_image: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert PIL image to bytes."""
    buf = BytesIO()
    pil_image.save(buf, format=fmt)
    return buf.getvalue()
