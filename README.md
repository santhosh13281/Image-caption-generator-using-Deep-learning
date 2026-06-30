# Image Caption Generator Using Deep Learning

An end-to-end AI application that generates natural language captions for images using **VGG16 + LSTM** on the **Flickr8k** dataset.

---

## Features

- Upload JPG/PNG images and generate AI captions
- VGG16 transfer learning + LSTM decoder
- Streamlit web UI with dark theme and sidebar navigation
- FastAPI REST backend
- SQLite prediction history
- BLEU evaluation and training metrics
- Modular, production-ready codebase

---

## Project Structure

```
Image-Caption-Generator/
├── app.py                    # Streamlit frontend (main entry)
├── requirements.txt
├── README.md
├── backend/
│   ├── preprocess.py         # Data preprocessing
│   ├── feature_extraction.py # VGG16 features
│   ├── model.py              # CNN-LSTM architecture
│   ├── train.py              # Training pipeline
│   ├── predict.py            # Caption generation
│   ├── database.py           # SQLite storage
│   ├── api.py                # FastAPI backend
│   ├── utils.py              # BLEU & helpers
│   └── config.py             # Configuration
├── frontend/
│   ├── styles.py             # UI styles
│   └── services.py           # Frontend services
├── dataset/
│   ├── Images/               # Flickr8k images
│   └── captions.txt          # Caption file
├── saved_models/             # Trained model artifacts
├── features/                 # Cached VGG16 features
├── database/                 # SQLite database
├── report/                   # Project documentation
└── screenshots/              # App screenshots
```

---

## Installation

### Prerequisites

- Python 3.11+
- ~4 GB disk space (dataset + model)
- GPU recommended for training

### Steps

```bash
# 1. Navigate to project
cd "Image caption generator using deep learning"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download Flickr8k dataset
# Place files in dataset/ (see dataset/README.md)
```

---

## Execution Guide

### 1. Preprocess Data

```bash
cd backend
python preprocess.py
```

### 2. Extract VGG16 Features

```bash
python feature_extraction.py
```

### 3. Train Model

```bash
python train.py
# Options: --epochs 20 --batch-size 64 --force-features
```

Outputs:
- `saved_models/image_caption_model.keras`
- `saved_models/tokenizer.pkl`
- `saved_models/training_loss.png`
- `saved_models/metrics.json`

### 4. Launch Web Application

```bash
cd ..
streamlit run app.py
```

Open `http://localhost:8501`

### 5. (Optional) Start FastAPI Backend

```bash
cd backend
python api.py
```

API docs: `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check (`{"status":"running"}`) |
| POST | `/generate-caption` | Upload image, get caption |
| GET | `/history` | Prediction history |
| DELETE | `/history/{id}` | Delete record |

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.11+ | Programming language |
| TensorFlow / Keras | Deep learning |
| VGG16 | CNN feature extractor |
| LSTM | Caption decoder |
| Streamlit | Web frontend |
| FastAPI | REST API |
| SQLite | Prediction history |
| Flickr8k | Training dataset |

---

## License

Educational and research use. Flickr8k has its own license terms.

*Built as an internship final project — Image Caption Generator Using Deep Learning*
