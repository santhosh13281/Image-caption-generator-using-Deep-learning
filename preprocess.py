"""
Unified data preprocessing module for the Flickr8k dataset.

Functions:
  - Load dataset (images + captions)
  - Clean captions, remove punctuation, lowercase
  - Add startseq / endseq tokens
  - Tokenize captions and build vocabulary
  - Save tokenizer
  - Split data 80% train / 10% val / 10% test
"""

import json
import logging
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from config import (
    CAPTIONS_FILE,
    DATASET_DIR,
    END_TOKEN,
    IMAGES_DIR,
    MAX_CAPTION_LENGTH,
    MIN_WORD_FREQ,
    RANDOM_SEED,
    START_TOKEN,
    TEST_IMAGES_FILE,
    TEST_SPLIT,
    TOKENIZER_PATH,
    TRAIN_IMAGES_FILE,
    TRAIN_SPLIT,
    UNKNOWN_TOKEN,
    VAL_IMAGES_FILE,
    VAL_SPLIT,
    VOCAB_PATH,
)
from preprocess_images import (
    list_dataset_images,
    normalize_pixels,
    preprocess_image,
    preprocess_pil_image,
    resize_image,
)

logger = logging.getLogger(__name__)

__all__ = [
    "load_dataset",
    "clean_caption",
    "load_raw_captions",
    "load_cleaned_captions",
    "build_vocabulary",
    "tokenize_caption",
    "save_tokenizer",
    "load_tokenizer",
    "save_vocabulary",
    "load_vocabulary",
    "load_train_val_test",
    "run_preprocessing",
    "run_caption_preprocessing",
    "preprocess_image",
    "preprocess_pil_image",
    "resize_image",
    "normalize_pixels",
    "list_dataset_images",
]


# ---------------------------------------------------------------------------
# Caption cleaning
# ---------------------------------------------------------------------------

def clean_caption(caption: str) -> str:
    """
    Clean a raw caption: lowercase, remove punctuation, wrap with start/end tokens.
    """
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return f"{START_TOKEN} {caption} {END_TOKEN}"


def load_dataset(captions_file: Path = CAPTIONS_FILE) -> Dict[str, List[str]]:
    """
    Load the full Flickr8k caption dataset.

    Returns:
        Dict mapping image filename → list of raw caption strings.
    """
    return load_raw_captions(captions_file)


def load_raw_captions(captions_file: Path = CAPTIONS_FILE) -> Dict[str, List[str]]:
    """
    Load captions from Flickr8k.token.txt or captions.txt.

    Format: image_name#caption_id<TAB>caption_text
    """
    if not captions_file.exists():
        raise FileNotFoundError(
            f"Captions file not found: {captions_file}\n"
            f"Place captions.txt or Flickr8k.token.txt in {DATASET_DIR}/"
        )

    captions: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                image_id, text = line.split("\t", 1)
                image_name = image_id.split("#")[0]
            elif "," in line and line.count(",") >= 1:
                parts = line.split(",", 1)
                image_name, text = parts[0].strip(), parts[1].strip()
            else:
                logger.warning("Malformed line skipped: %s", line[:60])
                continue
            captions.setdefault(image_name, []).append(text)

    logger.info("Loaded raw captions for %d images.", len(captions))
    return captions


def load_cleaned_captions(captions_file: Path = CAPTIONS_FILE) -> Dict[str, List[str]]:
    """Load captions and apply cleaning to each entry."""
    raw = load_raw_captions(captions_file)
    return {image: [clean_caption(cap) for cap in caps] for image, caps in raw.items()}


# ---------------------------------------------------------------------------
# Vocabulary & tokenization
# ---------------------------------------------------------------------------

def build_vocabulary(
    captions_dict: Dict[str, List[str]],
    min_freq: int = MIN_WORD_FREQ,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build word-to-index and index-to-word mappings."""
    counter: Counter = Counter()
    for caps in captions_dict.values():
        for cap in caps:
            counter.update(cap.split())

    word2idx: Dict[str, int] = {
        "<pad>": 0,
        START_TOKEN: 1,
        END_TOKEN: 2,
        UNKNOWN_TOKEN: 3,
    }
    idx = 4
    for word, count in counter.items():
        if word in word2idx or count < min_freq:
            continue
        word2idx[word] = idx
        idx += 1

    idx2word = {v: k for k, v in word2idx.items()}
    logger.info("Vocabulary size: %d (min_freq=%d)", len(word2idx), min_freq)
    return word2idx, idx2word


def tokenize_caption(caption: str, word2idx: Dict[str, int]) -> List[int]:
    """Convert a cleaned caption to a list of token indices."""
    unk = word2idx.get(UNKNOWN_TOKEN, 3)
    return [word2idx.get(word, unk) for word in caption.split()]


def captions_to_sequences(
    captions_dict: Dict[str, List[str]],
    word2idx: Dict[str, int],
    max_length: int = MAX_CAPTION_LENGTH,
) -> Dict[str, List[np.ndarray]]:
    """Tokenize and pad all captions per image."""
    result: Dict[str, List[np.ndarray]] = {}
    for image, caps in captions_dict.items():
        tokenized = [tokenize_caption(c, word2idx) for c in caps]
        padded = pad_sequences(tokenized, maxlen=max_length, padding="post")
        result[image] = [np.array(seq, dtype=np.int32) for seq in padded]
    return result


# ---------------------------------------------------------------------------
# Dataset splits (80 / 10 / 10)
# ---------------------------------------------------------------------------

def _load_split_file(path: Path) -> List[str]:
    """Load image filenames from a split file."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def _auto_split_images(image_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Create 80/10/10 train/val/test split when official split files are missing.
    """
    rng = random.Random(RANDOM_SEED)
    names = sorted(image_names)
    rng.shuffle(names)

    n = len(names)
    train_end = int(n * TRAIN_SPLIT)
    val_end = train_end + int(n * VAL_SPLIT)

    train = names[:train_end]
    val = names[train_end:val_end]
    test = names[val_end:]
    logger.info("Auto-split — train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def _filter_captions(
    captions_dict: Dict[str, List[str]],
    image_list: List[str],
) -> Dict[str, List[str]]:
    """Return only captions for images present in image_list."""
    return {img: captions_dict[img] for img in image_list if img in captions_dict}


def load_train_val_test() -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, List[str]],
]:
    """
    Load train / validation / test caption splits.

    Uses official Flickr8k split files when available; otherwise 80/10/10 auto-split.
    """
    all_captions = load_cleaned_captions()

    train_list = _load_split_file(TRAIN_IMAGES_FILE)
    val_list = _load_split_file(VAL_IMAGES_FILE)
    test_list = _load_split_file(TEST_IMAGES_FILE)

    if train_list and val_list and test_list:
        train = _filter_captions(all_captions, train_list)
        val = _filter_captions(all_captions, val_list)
        test = _filter_captions(all_captions, test_list)
    else:
        available = list(all_captions.keys())
        if IMAGES_DIR.exists():
            on_disk = {f.name for f in IMAGES_DIR.iterdir() if f.is_file()}
            available = [img for img in available if img in on_disk]
        train_list, val_list, test_list = _auto_split_images(available)
        train = _filter_captions(all_captions, train_list)
        val = _filter_captions(all_captions, val_list)
        test = _filter_captions(all_captions, test_list)

    logger.info("Splits — train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_tokenizer(word2idx: Dict[str, int], path: Path = TOKENIZER_PATH) -> None:
    """Save word2idx mapping as a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(word2idx, file)
    logger.info("Tokenizer saved to %s", path)


def load_tokenizer(path: Path = TOKENIZER_PATH) -> Dict[str, int]:
    """Load tokenizer from pickle file."""
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}. Run training first.")
    with open(path, "rb") as file:
        return pickle.load(file)


def save_vocabulary(
    word2idx: Dict[str, int],
    idx2word: Dict[int, str],
    path: Path = VOCAB_PATH,
) -> None:
    """Save vocabulary mappings to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "word2idx": word2idx,
        "idx2word": {str(k): v for k, v in idx2word.items()},
        "vocab_size": len(word2idx),
        "max_caption_length": MAX_CAPTION_LENGTH,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    logger.info("Vocabulary saved to %s", path)


def load_vocabulary(path: Path = VOCAB_PATH) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """Load vocabulary from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    word2idx = data["word2idx"]
    idx2word = {int(k): v for k, v in data["idx2word"].items()}
    return word2idx, idx2word, data["vocab_size"]


def run_preprocessing() -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[str, int],
    Dict[int, str],
]:
    """
    Execute full preprocessing pipeline.

    Returns:
        (train, val, test, word2idx, idx2word)
    """
    train, val, test = load_train_val_test()
    combined = {**train, **val, **test}
    word2idx, idx2word = build_vocabulary(combined)
    save_tokenizer(word2idx)
    save_vocabulary(word2idx, idx2word)
    return train, val, test, word2idx, idx2word


# Alias for backward compatibility
run_caption_preprocessing = run_preprocessing


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        train, val, test, w2i, i2w = run_preprocessing()
        print(f"Vocab: {len(w2i)} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
