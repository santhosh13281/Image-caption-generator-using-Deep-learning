"""
Utility functions for the Image Caption Generator.

Includes BLEU score calculation, metrics persistence, and helper utilities.
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    CAPTIONS_FILE,
    DATASET_DIR,
    IMAGES_DIR,
    METRICS_PATH,
    SAVED_MODELS_DIR,
    TRAINING_HISTORY_PATH,
    get_model_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BLEU Score
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> Counter:
    """Return n-gram counts for a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _bleu_score(reference: List[str], candidate: List[str], n: int) -> float:
    """Compute BLEU-n precision for a single reference/candidate pair."""
    if len(candidate) < n:
        return 0.0
    ref_ngrams = _ngrams(reference, n)
    cand_ngrams = _ngrams(candidate, n)
    if not cand_ngrams:
        return 0.0
    overlap = sum((ref_ngrams & cand_ngrams).values())
    return overlap / max(sum(cand_ngrams.values()), 1)


def compute_bleu(
    references: List[List[str]],
    candidates: List[List[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Compute corpus-level BLEU-1 through BLEU-4 scores.

    Args:
        references: List of reference token lists (one per sample).
        candidates: List of candidate token lists (one per sample).
        max_n: Maximum n-gram order.

    Returns:
        Dict with keys bleu1, bleu2, bleu3, bleu4.
    """
    if len(references) != len(candidates) or not references:
        return {f"bleu{i}": 0.0 for i in range(1, max_n + 1)}

    scores = {f"bleu{i}": 0.0 for i in range(1, max_n + 1)}
    for ref, cand in zip(references, candidates):
        for n in range(1, max_n + 1):
            scores[f"bleu{n}"] += _bleu_score(ref, cand, n)

    n_samples = len(references)
    return {k: round(v / n_samples, 4) for k, v in scores.items()}


def tokenize_for_bleu(text: str) -> List[str]:
    """Tokenize caption text for BLEU evaluation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------

def save_metrics(metrics: Dict, path: Path = METRICS_PATH) -> None:
    """Save evaluation metrics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics["saved_at"] = datetime.utcnow().isoformat()
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    logger.info("Metrics saved to %s", path)


def load_metrics(path: Path = METRICS_PATH) -> Dict:
    """Load saved metrics or return defaults."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def load_training_history(path: Path = TRAINING_HISTORY_PATH) -> Dict:
    """Load training history JSON."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def get_model_info() -> Dict:
    """
    Return model and dataset information for the UI.

    Returns:
        Dict with dataset name, image count, architecture, accuracy, BLEU, etc.
    """
    metrics = load_metrics()
    history = load_training_history()

    image_count = 0
    if IMAGES_DIR.exists():
        image_count = sum(
            1 for f in IMAGES_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    accuracy = None
    val_accuracy = None
    if history:
        if "accuracy" in history and history["accuracy"]:
            accuracy = round(history["accuracy"][-1], 4)
        if "val_accuracy" in history and history["val_accuracy"]:
            val_accuracy = round(history["val_accuracy"][-1], 4)

    train_loss = None
    val_loss = None
    if history:
        if "loss" in history and history["loss"]:
            train_loss = round(history["loss"][-1], 4)
        if "val_loss" in history and history["val_loss"]:
            val_loss = round(history["val_loss"][-1], 4)

    model_path = get_model_path()
    return {
        "dataset_name": "Flickr8k",
        "dataset_path": str(DATASET_DIR),
        "captions_file": str(CAPTIONS_FILE),
        "num_images": image_count,
        "cnn_model": "VGG16 (Transfer Learning)",
        "nlp_model": "LSTM Decoder",
        "embedding_dim": 256,
        "lstm_units": 512,
        "model_trained": model_path.exists(),
        "model_path": str(model_path) if model_path.exists() else "Not trained yet",
        "accuracy": accuracy,
        "val_accuracy": val_accuracy,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "bleu1": metrics.get("bleu1"),
        "bleu2": metrics.get("bleu2"),
        "bleu3": metrics.get("bleu3"),
        "bleu4": metrics.get("bleu4"),
    }


def format_timestamp(iso_str: str) -> Tuple[str, str]:
    """
    Split ISO timestamp into date and time strings.

    Returns:
        Tuple of (date_str, time_str).
    """
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    except (ValueError, AttributeError):
        return iso_str[:10] if len(iso_str) >= 10 else iso_str, ""


def validate_image_extension(filename: str, allowed: set) -> bool:
    """Check if file extension is allowed."""
    suffix = Path(filename).suffix.lower()
    return suffix in allowed
