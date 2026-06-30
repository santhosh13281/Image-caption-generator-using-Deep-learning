"""
CNN-LSTM model definition using the Keras Functional API.

Architecture:
  Image Input (4096) → Dense → Dropout
  Caption Input    → Embedding → LSTM (return_sequences)
  Merge (Add)      → TimeDistributed Dense (Softmax)
"""

import logging
from typing import Optional

from keras.layers import (
    Add,
    Dense,
    Dropout,
    Embedding,
    Input,
    LSTM,
    RepeatVector,
    TimeDistributed,
)
from keras.models import Model
from keras.optimizers import Adam

from config import (
    DROPOUT_RATE,
    EMBEDDING_DIM,
    LEARNING_RATE,
    LSTM_UNITS,
    MAX_CAPTION_LENGTH,
    VGG16_FEATURE_DIM,
)

logger = logging.getLogger(__name__)


def build_model(
    vocab_size: int,
    max_length: int = MAX_CAPTION_LENGTH,
    feature_dim: int = VGG16_FEATURE_DIM,
    embedding_dim: int = EMBEDDING_DIM,
    lstm_units: int = LSTM_UNITS,
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """
    Build the CNN-LSTM image captioning model.

    The image branch projects VGG16 features into an embedding space.
    The text branch embeds token sequences and processes them with LSTM.
    Both branches are merged via element-wise addition at each timestep
    before the final softmax vocabulary prediction.

    Args:
        vocab_size: Number of unique tokens in vocabulary.
        max_length: Maximum caption sequence length.
        feature_dim: VGG16 feature vector size (4096).
        embedding_dim: Word embedding dimension.
        lstm_units: LSTM hidden state size.
        dropout_rate: Dropout probability.

    Returns:
        Uncompiled Keras Model.
    """
    # --- Image feature branch ---
    image_input = Input(shape=(feature_dim,), name="image_features")
    image_dense = Dense(embedding_dim, activation="relu", name="image_dense")(image_input)
    image_dropout = Dropout(dropout_rate, name="image_dropout")(image_dense)

    # --- Caption sequence branch ---
    caption_input = Input(shape=(max_length,), name="caption_input")
    caption_embed = Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=True,
        name="caption_embedding",
    )(caption_input)
    caption_lstm = LSTM(
        lstm_units,
        return_sequences=True,
        name="caption_lstm",
    )(caption_embed)
    caption_dropout = Dropout(dropout_rate, name="caption_dropout")(caption_lstm)

    # --- Merge image context with language model at every timestep ---
    image_repeated = RepeatVector(max_length, name="repeat_image")(image_dropout)

    if lstm_units != embedding_dim:
        image_repeated = TimeDistributed(
            Dense(lstm_units, activation="relu"),
            name="align_image_dim",
        )(image_repeated)

    merged = Add(name="merge_branches")([image_repeated, caption_dropout])

    # --- Output: softmax over vocabulary at each timestep ---
    output = TimeDistributed(
        Dense(vocab_size, activation="softmax"),
        name="word_output",
    )(merged)

    model = Model(
        inputs=[image_input, caption_input],
        outputs=output,
        name="image_caption_model",
    )

    logger.info(
        "Model built — vocab=%d, max_len=%d, params=%d",
        vocab_size,
        max_length,
        model.count_params(),
    )
    return model


def compile_model(
    model: Model,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """
    Compile model with Adam optimizer and categorical cross-entropy loss.

    Uses sparse_categorical_crossentropy because target sequences are
    integer token indices (mathematically equivalent to one-hot CE).

    Args:
        model: Uncompiled Keras model.
        learning_rate: Adam learning rate.

    Returns:
        Compiled model.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info("Model compiled — Adam(lr=%.4f), sparse_categorical_crossentropy", learning_rate)
    return model


def get_model(
    vocab_size: int,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """Build and compile model in one step."""
    model = build_model(vocab_size)
    return compile_model(model, learning_rate)
