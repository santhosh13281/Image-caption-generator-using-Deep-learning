"""
VGG16 feature extraction module.

Loads a pre-trained VGG16 model, removes the classification head,
extracts 4096-dimensional feature vectors, and caches them as pickle files.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from PIL import Image
from tqdm import tqdm

from config import FEATURES_DIR, FEATURES_PKL, IMAGES_DIR, VGG16_FEATURE_DIM
from preprocess_images import preprocess_image, preprocess_pil_image

logger = logging.getLogger(__name__)


class VGG16FeatureExtractor:
    """
    Extract 4096-dimensional features from images using pre-trained VGG16.

    The model is frozen (transfer learning) — only the fc1 layer output
    is used as the image representation for the LSTM decoder.
    """

    def __init__(self) -> None:
        self.model = self._build_extractor()
        logger.info("VGG16 extractor ready (output dim=%d).", VGG16_FEATURE_DIM)

    @staticmethod
    def _build_extractor() -> Model:
        """
        Build VGG16 truncated at the fc1 layer (4096 units).

        Returns:
            Keras Model: (224, 224, 3) → (4096,)
        """
        base = VGG16(weights="imagenet", include_top=True)
        extractor = Model(
            inputs=base.input,
            outputs=base.get_layer("fc1").output,
            name="vgg16_feature_extractor",
        )
        extractor.trainable = False
        return extractor

    def extract_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract features from a preprocessed image array.

        Args:
            image_array: Shape (1, 224, 224, 3), VGG16-normalized.

        Returns:
            1-D float32 array of shape (4096,).
        """
        features = self.model.predict(image_array, verbose=0)
        return features.flatten().astype(np.float32)

    def extract_from_path(self, image_path: Union[str, Path]) -> np.ndarray:
        """Extract features from an image file path."""
        array = preprocess_image(image_path)
        return self.extract_from_array(array)

    def extract_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """Extract features from an in-memory PIL image."""
        array = preprocess_pil_image(pil_image)
        return self.extract_from_array(array)


# ---------------------------------------------------------------------------
# Batch extraction & caching
# ---------------------------------------------------------------------------

def extract_features_batch(
    image_names: List[str],
    images_dir: Path = IMAGES_DIR,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract VGG16 features for a list of image filenames.

    Args:
        image_names: List of image filenames (not full paths).
        images_dir: Directory containing images.
        show_progress: Show tqdm progress bar.

    Returns:
        Dict mapping image filename → 4096-dim feature vector.
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    extractor = VGG16FeatureExtractor()
    features: Dict[str, np.ndarray] = {}
    iterator = tqdm(image_names, desc="Extracting features") if show_progress else image_names

    for name in iterator:
        path = images_dir / name
        try:
            features[name] = extractor.extract_from_path(path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipped %s: %s", name, exc)

    logger.info("Extracted features for %d / %d images.", len(features), len(image_names))
    return features


def save_features(
    features: Dict[str, np.ndarray],
    path: Path = FEATURES_PKL,
) -> None:
    """Persist feature dictionary to a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(features, file)
    logger.info("Features cached to %s (%d images).", path, len(features))


def load_features(path: Path = FEATURES_PKL) -> Dict[str, np.ndarray]:
    """
    Load cached features from pickle.

    Raises:
        FileNotFoundError: If cache file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Features cache not found: {path}. Run feature extraction first."
        )
    with open(path, "rb") as file:
        features = pickle.load(file)
    logger.info("Loaded %d cached feature vectors from %s.", len(features), path)
    return features


def extract_and_cache(
    image_names: List[str],
    output_path: Path = FEATURES_PKL,
    force_recompute: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Extract features with automatic caching.

    Loads from cache unless force_recompute=True or cache is missing.

    Args:
        image_names: Image filenames to process.
        output_path: Pickle cache path.
        force_recompute: Ignore existing cache.

    Returns:
        Feature dictionary.
    """
    if not force_recompute and output_path.exists():
        cached = load_features(output_path)
        missing = [n for n in image_names if n not in cached]
        if not missing:
            logger.info("All features loaded from cache.")
            return {k: cached[k] for k in image_names if k in cached}
        logger.info("%d images missing from cache — extracting.", len(missing))
        new_features = extract_features_batch(missing)
        cached.update(new_features)
        save_features(cached, output_path)
        return cached

    features = extract_features_batch(image_names)
    save_features(features, output_path)
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from preprocess import load_train_val_test

    try:
        train, val, test = load_train_val_test()
        all_images = list(set(train) | set(val) | set(test))
        features = extract_and_cache(all_images)
        print(f"Done. {len(features)} feature vectors cached.")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
