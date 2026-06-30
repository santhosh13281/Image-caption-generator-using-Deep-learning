"""
Efficient batch data generator for CNN-LSTM caption training.

Creates (image_feature, input_sequence) → output_sequence pairs using
pre-extracted VGG16 features to avoid redundant CNN forward passes.
"""

import logging
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence

from config import BATCH_SIZE, MAX_CAPTION_LENGTH
from preprocess import tokenize_caption

logger = logging.getLogger(__name__)


def build_training_samples(
    captions_dict: Dict[str, List[str]],
    features: Dict[str, np.ndarray],
    word2idx: Dict[str, int],
    max_length: int = MAX_CAPTION_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
  Build flat training arrays from captions and pre-extracted features.

    For teacher forcing:
      - input_sequence  = tokens[:-1]  (padded)
      - output_sequence = tokens[1:]     (padded, used as labels)

    Args:
        captions_dict: Image → cleaned captions.
        features: Image → 4096-dim feature vector.
        word2idx: Vocabulary mapping.
        max_length: Max sequence length.

    Returns:
        X_img:    (N, 4096) float32
        X_text:   (N, max_length) int32
        y:        (N, max_length) int32  — integer labels per timestep
    """
    img_list: List[np.ndarray] = []
    in_list: List[np.ndarray] = []
    out_list: List[np.ndarray] = []
    skipped = 0

    for image_name, captions in captions_dict.items():
        if image_name not in features:
            skipped += 1
            continue

        feat = features[image_name]
        for caption in captions:
            tokens = tokenize_caption(caption, word2idx)
            if len(tokens) < 2:
                continue

            in_seq = pad_sequences([tokens[:-1]], maxlen=max_length, padding="post")[0]
            out_seq = pad_sequences([tokens[1:]], maxlen=max_length, padding="post")[0]

            img_list.append(feat)
            in_list.append(in_seq)
            out_list.append(out_seq)

    if skipped:
        logger.warning("%d images had no cached features.", skipped)

    if not img_list:
        raise ValueError("No training samples could be created. Check features and captions.")

    X_img = np.array(img_list, dtype=np.float32)
    X_text = np.array(in_list, dtype=np.int32)
    y = np.array(out_list, dtype=np.int32)

    logger.info("Built %d training samples.", len(X_img))
    return X_img, X_text, y


class CaptionDataGenerator(Sequence):
    """
    Keras Sequence generator for memory-efficient batch training.

    Shuffles indices each epoch and yields mini-batches of
    (image_features, input_sequences) → output_sequences.

    Args:
        X_img:    Image feature array (N, 4096).
        X_text:   Input caption sequences (N, max_length).
        y:        Target sequences (N, max_length).
        batch_size: Mini-batch size.
        shuffle:  Shuffle data at end of each epoch.
    """

    def __init__(
        self,
        X_img: np.ndarray,
        X_text: np.ndarray,
        y: np.ndarray,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
    ) -> None:
        self.X_img = X_img
        self.X_text = X_text
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X_img))
        self.on_epoch_end()

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.X_img) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Return one mini-batch.

        Returns:
            ([X_img_batch, X_text_batch], y_batch)
        """
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.X_img))
        batch_idx = self.indices[start:end]

        X_img_batch = self.X_img[batch_idx]
        X_text_batch = self.X_text[batch_idx]
        y_batch = self.y[batch_idx]

        return [X_img_batch, X_text_batch], y_batch

    def on_epoch_end(self) -> None:
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_generators(
    train_captions: Dict[str, List[str]],
    val_captions: Dict[str, List[str]],
    features: Dict[str, np.ndarray],
    word2idx: Dict[str, int],
    batch_size: int = BATCH_SIZE,
) -> Tuple[CaptionDataGenerator, CaptionDataGenerator]:
    """
    Create training and validation data generators.

    Args:
        train_captions: Training split captions.
        val_captions: Validation split captions.
        features: Cached VGG16 features.
        word2idx: Vocabulary mapping.
        batch_size: Mini-batch size.

    Returns:
        Tuple of (train_generator, val_generator).
    """
    X_img_tr, X_txt_tr, y_tr = build_training_samples(train_captions, features, word2idx)
    X_img_val, X_txt_val, y_val = build_training_samples(val_captions, features, word2idx)

    train_gen = CaptionDataGenerator(X_img_tr, X_txt_tr, y_tr, batch_size=batch_size)
    val_gen = CaptionDataGenerator(X_img_val, X_txt_val, y_val, batch_size=batch_size, shuffle=False)

    return train_gen, val_gen


def create_static_arrays(
    captions_dict: Dict[str, List[str]],
    features: Dict[str, np.ndarray],
    word2idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper returning static numpy arrays (for model.fit without generator).

    Returns:
        (X_img, X_text, y)
    """
    return build_training_samples(captions_dict, features, word2idx)
