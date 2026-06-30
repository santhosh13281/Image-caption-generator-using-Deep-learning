"""
Training pipeline for the CNN-LSTM Image Caption Generator.

Steps:
  1. Preprocess captions and build vocabulary
  2. Extract / load cached VGG16 features
  3. Build training arrays via data generator
  4. Train with early stopping and checkpointing
  5. Save model (image_caption_model.h5) and tokenizer.pkl
  6. Plot loss and accuracy curves
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import (
    ACCURACY_PLOT_PATH,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LOGS_DIR,
    LOSS_PLOT_PATH,
    MODEL_H5_PATH,
    MODEL_KERAS_PATH,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    SAVED_MODELS_DIR,
    TOKENIZER_PATH,
    TRAINING_HISTORY_PATH,
)
from data_generator import create_static_arrays
from feature_extraction import extract_and_cache
from model import build_model, compile_model
from preprocess import run_preprocessing
from utils import compute_bleu, save_metrics, tokenize_for_bleu

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "training.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def plot_training_curves(history: dict) -> None:
    """
    Save loss and accuracy plots to saved_models/.

    Args:
        history: Keras History.history dictionary.
    """
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    # Loss plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["loss"], "b-o", label="Train Loss", markersize=4)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()
    logger.info("Loss plot saved → %s", LOSS_PLOT_PATH)

    # Accuracy plot
    if "accuracy" in history:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, history["accuracy"], "b-o", label="Train Accuracy", markersize=4)
        if "val_accuracy" in history:
            ax.plot(epochs, history["val_accuracy"], "r-o", label="Val Accuracy", markersize=4)
        ax.set_title("Training & Validation Accuracy", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ACCURACY_PLOT_PATH, dpi=150)
        plt.close()
        logger.info("Accuracy plot saved → %s", ACCURACY_PLOT_PATH)


def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    force_features: bool = False,
) -> dict:
    """
    Run the full training pipeline.

    Args:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        force_features: Re-extract VGG16 features.

    Returns:
        Training history dictionary.
    """
    logger.info("=" * 60)
    logger.info("Image Caption Generator — Training")
    logger.info("=" * 60)

    # Step 1: Caption preprocessing
    logger.info("[1/4] Preprocessing captions...")
    train_caps, val_caps, test_caps, word2idx, idx2word = run_preprocessing()
    vocab_size = len(word2idx)

    # Step 2: Feature extraction
    logger.info("[2/4] Loading / extracting VGG16 features...")
    all_images = list(set(train_caps) | set(val_caps) | set(test_caps))
    features = extract_and_cache(all_images, force_recompute=force_features)

    # Step 3: Build training data
    logger.info("[3/4] Building training arrays...")
    X_img_tr, X_txt_tr, y_tr = create_static_arrays(train_caps, features, word2idx)
    X_img_val, X_txt_val, y_val = create_static_arrays(val_caps, features, word2idx)
    logger.info("Train samples: %d | Val samples: %d", len(X_img_tr), len(X_img_val))

    # Step 4: Build and train model
    logger.info("[4/4] Training CNN-LSTM model...")
    model = build_model(vocab_size)
    model = compile_model(model, learning_rate)

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            str(MODEL_KERAS_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history_obj = model.fit(
        [X_img_tr, X_txt_tr],
        y_tr,
        validation_data=([X_img_val, X_txt_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Persist artifacts
    model.save(str(MODEL_KERAS_PATH))
    logger.info("Model saved → %s", MODEL_KERAS_PATH)
    logger.info("Tokenizer saved → %s", TOKENIZER_PATH)

    history = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}
    with open(TRAINING_HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    plot_training_curves(history)

    # BLEU evaluation on test set (sample for speed)
    logger.info("Evaluating BLEU scores on test set...")
    bleu_metrics = _evaluate_bleu(model, test_caps, features, word2idx)
    save_metrics({
        **bleu_metrics,
        "final_train_loss": history["loss"][-1] if history.get("loss") else None,
        "final_val_loss": history["val_loss"][-1] if history.get("val_loss") else None,
        "final_train_accuracy": history["accuracy"][-1] if history.get("accuracy") else None,
        "final_val_accuracy": history["val_accuracy"][-1] if history.get("val_accuracy") else None,
    })
    logger.info("BLEU scores: %s", bleu_metrics)

    logger.info("Training complete.")
    return history


def _evaluate_bleu(model, test_caps, features, word2idx, max_samples: int = 100) -> dict:
    """Compute BLEU scores on a sample of test images."""
    from predict import CaptionPredictor

    predictor = CaptionPredictor(model_path=MODEL_KERAS_PATH)
    references, candidates = [], []
    count = 0

    for image_name, caps in test_caps.items():
        if image_name not in features or count >= max_samples:
            continue
        ref_tokens = tokenize_for_bleu(caps[0].replace("startseq ", "").replace(" endseq", ""))
        feat = features[image_name]
        caption, _, _ = predictor.generate_caption(feat)
        cand_tokens = tokenize_for_bleu(caption)
        references.append(ref_tokens)
        candidates.append(cand_tokens)
        count += 1

    return compute_bleu(references, candidates) if references else {
        "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Image Caption Generator")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    try:
        train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            force_features=args.force_features,
        )
    except FileNotFoundError as exc:
        logger.error("Dataset error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
