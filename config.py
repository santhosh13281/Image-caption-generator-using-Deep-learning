"""
Central configuration for the Image Caption Generator.
Supports Flickr8k official layout and simplified dataset/Images + captions.txt layout.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent

# Dataset — project root (preferred) with fallback to backend/dataset
_ROOT_DATASET = PROJECT_ROOT / "dataset"
_BACKEND_DATASET = BACKEND_ROOT / "dataset"

if (_ROOT_DATASET / "Images").exists() or (_ROOT_DATASET / "captions.txt").exists():
    DATASET_DIR = _ROOT_DATASET
elif (_ROOT_DATASET / "Flicker8k_Dataset").exists() or (_ROOT_DATASET / "Flickr8k.token.txt").exists():
    DATASET_DIR = _ROOT_DATASET
elif _BACKEND_DATASET.exists():
    DATASET_DIR = _BACKEND_DATASET
else:
    DATASET_DIR = _ROOT_DATASET

# Images directory (auto-detect layout)
if (DATASET_DIR / "Images").exists():
    IMAGES_DIR = DATASET_DIR / "Images"
else:
    IMAGES_DIR = DATASET_DIR / "Flicker8k_Dataset"

# Captions file (auto-detect)
if (DATASET_DIR / "captions.txt").exists():
    CAPTIONS_FILE = DATASET_DIR / "captions.txt"
else:
    CAPTIONS_FILE = DATASET_DIR / "Flickr8k.token.txt"

TRAIN_IMAGES_FILE = DATASET_DIR / "Flickr_8k.trainImages.txt"
VAL_IMAGES_FILE = DATASET_DIR / "Flickr_8k.devImages.txt"
TEST_IMAGES_FILE = DATASET_DIR / "Flickr_8k.testImages.txt"

FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
LOGS_DIR = PROJECT_ROOT / "logs"
DATABASE_DIR = PROJECT_ROOT / "database"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
REPORT_DIR = PROJECT_ROOT / "report"

DB_PATH = DATABASE_DIR / "captions.db"

# Artifacts
FEATURES_PKL = FEATURES_DIR / "image_features.pkl"
MODEL_KERAS_PATH = SAVED_MODELS_DIR / "image_caption_model.keras"
MODEL_H5_PATH = SAVED_MODELS_DIR / "image_caption_model.h5"
TOKENIZER_PATH = SAVED_MODELS_DIR / "tokenizer.pkl"
VOCAB_PATH = SAVED_MODELS_DIR / "vocab.json"
TRAINING_HISTORY_PATH = SAVED_MODELS_DIR / "training_history.json"
METRICS_PATH = SAVED_MODELS_DIR / "metrics.json"
LOSS_PLOT_PATH = SAVED_MODELS_DIR / "training_loss.png"
ACCURACY_PLOT_PATH = SAVED_MODELS_DIR / "training_accuracy.png"

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
VGG16_FEATURE_DIM = 4096

# ---------------------------------------------------------------------------
# Caption settings
# ---------------------------------------------------------------------------
START_TOKEN = "startseq"
END_TOKEN = "endseq"
UNKNOWN_TOKEN = "unk"
PAD_TOKEN = "<pad>"
MIN_WORD_FREQ = 5
MAX_CAPTION_LENGTH = 40

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 256
LSTM_UNITS = 512
DROPOUT_RATE = 0.5

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
MAX_GENERATION_LENGTH = 40

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_model_path() -> Path:
    """Return the trained model path (.keras preferred, .h5 fallback)."""
    if MODEL_KERAS_PATH.exists():
        return MODEL_KERAS_PATH
    return MODEL_H5_PATH


def ensure_directories() -> None:
    """Create required directories if they do not exist."""
    for directory in (
        DATASET_DIR,
        IMAGES_DIR,
        FEATURES_DIR,
        MODELS_DIR,
        SAVED_MODELS_DIR,
        UPLOADS_DIR,
        LOGS_DIR,
        DATABASE_DIR,
        SCREENSHOTS_DIR,
        REPORT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
