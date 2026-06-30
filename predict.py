"""
Caption prediction module.

Loads the trained model and tokenizer, extracts VGG16 features from
an input image, and generates a caption word-by-word until endseq.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from keras.models import load_model
from PIL import Image

from config import (
    END_TOKEN,
    MAX_CAPTION_LENGTH,
    MAX_GENERATION_LENGTH,
    START_TOKEN,
    TOKENIZER_PATH,
    UNKNOWN_TOKEN,
    get_model_path,
)
from feature_extraction import VGG16FeatureExtractor
from preprocess import load_tokenizer

logger = logging.getLogger(__name__)


class CaptionPredictor:
    """
    End-to-end caption generation for unseen images.

    Usage:
        predictor = CaptionPredictor()
        result = predictor.predict("path/to/image.jpg")
        print(result["caption"])
    """

    def __init__(
        self,
        model_path: Union[str, Path] = None,
        tokenizer_path: Union[str, Path] = TOKENIZER_PATH,
    ) -> None:
        """
        Load model, tokenizer, and VGG16 feature extractor.

        Raises:
            FileNotFoundError: If model or tokenizer files are missing.
        """
        model_path = Path(model_path or get_model_path())
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\nRun: python train.py"
            )

        self.model = load_model(str(model_path))
        self.word2idx = load_tokenizer(Path(tokenizer_path))
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.max_length = MAX_CAPTION_LENGTH
        self.extractor = VGG16FeatureExtractor()

        logger.info("CaptionPredictor ready (vocab=%d).", self.vocab_size)

    def _word_to_idx(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx.get(UNKNOWN_TOKEN, 3))

    def _idx_to_word(self, idx: int) -> str:
        return self.idx2word.get(idx, UNKNOWN_TOKEN)

    def generate_caption(
        self,
        image_features: np.ndarray,
    ) -> Tuple[str, float, List[float]]:
        """
        Generate a caption from a 4096-dim feature vector using greedy decoding.

        Generates one word at a time until endseq or max length is reached.

        Args:
            image_features: 1-D array of shape (4096,).

        Returns:
            Tuple of (caption_string, average_confidence, step_confidences).
        """
        start_idx = self._word_to_idx(START_TOKEN)
        end_idx = self._word_to_idx(END_TOKEN)

        in_text = [start_idx]
        words: List[str] = []
        confidences: List[float] = []

        for _ in range(MAX_GENERATION_LENGTH):
            # Build padded input sequence
            sequence = np.zeros((1, self.max_length), dtype=np.int32)
            for i, token in enumerate(in_text):
                if i < self.max_length:
                    sequence[0, i] = token

            img_input = np.expand_dims(image_features, axis=0).astype(np.float32)
            predictions = self.model.predict([img_input, sequence], verbose=0)

            pos = min(len(in_text) - 1, self.max_length - 1)
            word_probs = predictions[0, pos, :]

            next_idx = int(np.argmax(word_probs))
            confidence = float(word_probs[next_idx])
            confidences.append(confidence)

            # Stop when endseq is predicted
            if next_idx == end_idx:
                break

            word = self._idx_to_word(next_idx)
            if word not in (START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, "<pad>"):
                words.append(word)

            in_text.append(next_idx)

        caption = " ".join(words)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        return caption, avg_confidence, confidences

    def predict_from_path(self, image_path: Union[str, Path]) -> Dict:
        """
        Generate caption for an image file.

        Args:
            image_path: Path to image.

        Returns:
            Dict with keys: caption, confidence, step_confidences.
        """
        features = self.extractor.extract_from_path(image_path)
        caption, confidence, step_confs = self.generate_caption(features)
        return {
            "caption": caption,
            "confidence": confidence,
            "step_confidences": step_confs,
        }

    def predict_from_pil(self, pil_image: Image.Image) -> Dict:
        """
        Generate caption from a PIL Image (for API / Streamlit).

        Args:
            pil_image: PIL Image object.

        Returns:
            Dict with keys: caption, confidence, step_confidences.
        """
        features = self.extractor.extract_from_pil(pil_image)
        caption, confidence, step_confs = self.generate_caption(features)
        return {
            "caption": caption,
            "confidence": confidence,
            "step_confidences": step_confs,
        }


# Module-level singleton for API reuse
_predictor: Optional[CaptionPredictor] = None


def get_predictor() -> CaptionPredictor:
    """
    Return a cached CaptionPredictor instance (lazy-loaded).

    Raises:
        FileNotFoundError: If model has not been trained yet.
    """
    global _predictor
    if _predictor is None:
        _predictor = CaptionPredictor()
    return _predictor


def predict(image_path: Union[str, Path]) -> str:
    """
    Convenience function — returns caption string only.

    Args:
        image_path: Path to image file.

    Returns:
        Generated caption string.
    """
    return get_predictor().predict_from_path(image_path)["caption"]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    try:
        result = get_predictor().predict_from_path(sys.argv[1])
        print(f"Caption:    {result['caption']}")
        print(f"Confidence: {result['confidence']:.2%}")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
