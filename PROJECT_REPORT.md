# Image Caption Generator — Project Report

## Abstract

This project presents an end-to-end **Image Caption Generator** that automatically produces natural language descriptions for uploaded images. The system combines **Computer Vision** (VGG16 CNN) and **Natural Language Processing** (LSTM) using the **Flickr8k** dataset. A Streamlit web interface and FastAPI backend provide production-ready deployment.

---

## Problem Statement

Images are a universal communication medium but lack textual metadata for search, accessibility, and content management. Manual annotation of large image collections is costly and slow. Automated image captioning bridges this gap by generating human-readable descriptions using deep learning.

---

## Objectives

1. Implement VGG16 transfer learning for image feature extraction
2. Build an LSTM-based caption decoder
3. Preprocess Flickr8k (cleaning, tokenization, vocabulary)
4. Train with early stopping, checkpointing, and learning rate scheduling
5. Evaluate using BLEU scores
6. Deploy via Streamlit frontend and FastAPI REST API
7. Store prediction history in SQLite

---

## Literature Review

Image captioning has evolved from template-based methods to encoder-decoder architectures. **Show and Tell** (Vinyals et al., 2015) pioneered CNN-RNN models. **VGG16** (Simonyan & Zisserman, 2014) provides robust visual features. **LSTM** networks handle sequential language generation. Flickr8k remains a standard benchmark for captioning research.

---

## Methodology

### Data Preprocessing
- Resize images to 224×224
- VGG16 normalization
- Caption cleaning: lowercase, punctuation removal
- Add `startseq` / `endseq` tokens
- Tokenization and vocabulary (min frequency = 5)
- 80/10/10 train/validation/test split

### Feature Extraction
- Pre-trained VGG16 (ImageNet weights, frozen)
- Extract 4096-dim features from fc1 layer
- Cache features to pickle for fast retraining

### Model Architecture
```
Image Features (4096) → Dense(256) → Dropout
Caption Input → Embedding(256) → LSTM(512)
Merge (Add) → TimeDistributed Dense → Softmax
```

### Training
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

### Inference
- Greedy word-by-word decoding until `endseq`
- Confidence from softmax probabilities

---

## System Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Streamlit  │────▶│   FastAPI    │────▶│  Predictor   │
│   Frontend   │     │   Backend    │     │  (VGG+LSTM)  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                     │
                            ▼                     ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   SQLite     │     │ Saved Model  │
                     │   Database   │     │  + Tokenizer │
                     └──────────────┘     └──────────────┘
```

---

## Algorithm

**Training:** Load captions → Clean → Tokenize → Extract VGG16 features → Create (image, input_seq) → output_seq pairs → Train LSTM decoder

**Inference:** Upload image → Preprocess → VGG16 features → Initialize with startseq → Predict next word → Repeat until endseq → Return caption

---

## Modules

| Module | File | Purpose |
|--------|------|---------|
| Preprocessing | `backend/preprocess.py` | Dataset loading, caption cleaning, tokenization |
| Feature Extraction | `backend/feature_extraction.py` | VGG16 feature extraction and caching |
| Model | `backend/model.py` | CNN-LSTM architecture |
| Training | `backend/train.py` | Full training pipeline |
| Prediction | `backend/predict.py` | Caption generation |
| Database | `backend/database.py` | SQLite history |
| API | `backend/api.py` | FastAPI REST endpoints |
| Utils | `backend/utils.py` | BLEU scores, metrics |
| Frontend | `app.py` | Streamlit web application |

---

## Dataset Description

**Flickr8k** contains 8,091 images with 5 English captions each (40,455 total). Images depict diverse everyday scenes. Standard splits: 6,000 train, 1,000 validation, 1,000 test.

---

## Flowchart

```
Download Dataset → Preprocess Captions → Extract VGG16 Features
        → Build Model → Train → Save Model & Tokenizer
        → Evaluate BLEU → Deploy Streamlit App → User Uploads Image → Generate Caption
```

---

## Results

Expected metrics on Flickr8k (CNN-LSTM baseline):
- BLEU-1: ~0.50–0.55
- BLEU-4: ~0.12–0.18
- Training accuracy: ~65–75% (validation)

---

## Advantages

- Transfer learning reduces training time
- Feature caching enables fast experiment iteration
- Modular, beginner-friendly codebase
- Full-stack deployment (Streamlit + FastAPI + SQLite)

---

## Applications

- Accessibility for visually impaired users
- Social media auto-tagging
- Image search and content management
- E-commerce product descriptions

---

## Future Scope

- Attention mechanisms
- Transformer decoders
- MS COCO training
- Multi-language support
- Mobile/edge deployment

---

## Conclusion

This project delivers a complete, production-ready image captioning system suitable for internship final project submission. It demonstrates CNN-RNN integration, proper ML engineering practices, and modern web deployment.

---

## References

1. Vinyals, O., et al. (2015). Show and Tell: A Neural Image Caption Generator. CVPR.
2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR.
3. Hodosh, M., et al. (2013). Framing Image Description as a Ranking Task. EMNLP.
4. Flickr8k Dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k
