"""
Image preprocessing module for the Flickr8k dataset.

Responsibilities:
  - Load images from disk
  - Resize to 224x224 (VGG16 input size)
  - Normalize pixel values using VGG16 preprocess_input
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

from config import IMAGES_DIR, IMAGE_CHANNELS, IMAGE_SIZE

logger = logging.getLogger(__name__)


def load_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> Image.Image:
    """
    Load an image file as a PIL Image in RGB mode.

    Args:
        image_path: Path to the image file.
        target_size: Unused at load time; kept for API consistency.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be opened.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception as exc:
        raise ValueError(f"Failed to load image '{path}': {exc}") from exc


def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> Image.Image:
    """
    Resize a PIL image to the target dimensions.

    Args:
        image: Source PIL Image.
        target_size: (width, height) tuple.

    Returns:
        Resized PIL Image.
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)


def normalize_pixels(image_array: np.ndarray) -> np.ndarray:
    """
    Apply VGG16-specific pixel normalization.

    Converts RGB values to the mean-subtracted BGR layout expected by VGG16.

    Args:
        image_array: Float or uint8 array of shape (H, W, 3) or (1, H, W, 3).

    Returns:
        Normalized float32 array with the same batch structure.
    """
    if image_array.ndim == 3:
        image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array.astype(np.float32))


def preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Full image preprocessing pipeline: load → resize → array → normalize.

    Args:
        image_path: Path to image file.
        target_size: Target (width, height).

    Returns:
        Normalized array of shape (1, 224, 224, 3).
    """
    path = Path(image_path)
    image = load_img(str(path), target_size=target_size)
    array = img_to_array(image)
    array = np.expand_dims(array, axis=0)
    return normalize_pixels(array)


def preprocess_pil_image(
    pil_image: Image.Image,
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Preprocess an in-memory PIL image (used by API uploads).

    Args:
        pil_image: PIL Image object.
        target_size: Target (width, height).

    Returns:
        Normalized array of shape (1, 224, 224, 3).
    """
    image = pil_image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    array = img_to_array(image)
    array = np.expand_dims(array, axis=0)
    return normalize_pixels(array)


def load_and_preprocess_batch(
    image_paths: List[Union[str, Path]],
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load and preprocess a batch of images.

    Skips images that fail to load and logs warnings.

    Args:
        image_paths: List of image file paths.
        target_size: Target (width, height).

    Returns:
        Tuple of (batch_array, successful_filenames).
        batch_array shape: (N, H, W, 3).
    """
    arrays: List[np.ndarray] = []
    names: List[str] = []

    for path in image_paths:
        try:
            array = preprocess_image(path, target_size)
            arrays.append(array[0])
            names.append(Path(path).name)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping image %s: %s", path, exc)

    if not arrays:
        raise ValueError("No images could be loaded from the provided paths.")

    return np.array(arrays, dtype=np.float32), names


def list_dataset_images(images_dir: Path = IMAGES_DIR) -> List[str]:
    """
    List all image filenames in the Flickr8k images directory.

    Args:
        images_dir: Path to Flicker8k_Dataset folder.

    Returns:
        Sorted list of image filenames.

    Raises:
        FileNotFoundError: If the images directory does not exist.
    """
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            "Place Flickr8k images in backend/dataset/Flicker8k_Dataset/"
        )

    extensions = {".jpg", ".jpeg", ".png"}
    images = [
        f.name
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]
    logger.info("Found %d images in %s", len(images), images_dir)
    return sorted(images)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        images = list_dataset_images()
        print(f"Dataset contains {len(images)} images.")
        if images:
            sample = IMAGES_DIR / images[0]
            processed = preprocess_image(sample)
            print(f"Sample shape: {processed.shape}, dtype: {processed.dtype}")
    except FileNotFoundError as exc:
        logger.error("%s", exc)
