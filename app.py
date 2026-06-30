"""
Image Caption Generator — Streamlit Web Application

Run:
    streamlit run app.py

Optional FastAPI backend (for REST API access):
    cd backend && python api.py
"""

import os
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from frontend.services import (  # noqa: E402
    clear_all_history,
    delete_history_record,
    generate_caption_api,
    generate_caption_direct,
    get_api_health,
    get_history,
    get_model_info,
    image_to_bytes,
    is_model_available,
    save_prediction,
    validate_upload,
)
from frontend.styles import DARK_THEME_CSS  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "caption_result" not in st.session_state:
    st.session_state.caption_result = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "use_api" not in st.session_state:
    st.session_state.use_api = False


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📷 Caption AI")
    st.markdown("---")

    pages = {
        "🏠 Home": "Home",
        "🖼️ Upload & Generate": "Upload",
        "📜 Prediction History": "History",
        "🧠 Model Information": "Model Info",
        "ℹ️ About Project": "About",
    }

    for label, key in pages.items():
        if st.button(label, use_container_width=True, key=f"nav_{key}"):
            st.session_state.page = key

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.session_state.use_api = st.toggle(
        "Use FastAPI Backend",
        value=st.session_state.use_api,
        help="Enable if running `python backend/api.py` separately",
    )
    if st.session_state.use_api:
        api_url = st.text_input("API URL", value=API_URL)
        health = get_api_health(api_url)
        if health.get("status") == "running":
            st.markdown('<span class="status-badge status-online">● API Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-offline">● API Offline</span>', unsafe_allow_html=True)
    else:
        api_url = API_URL

    st.markdown("---")
    model_ready = is_model_available()
    if model_ready:
        st.success("✅ Model Ready")
    else:
        st.warning("⚠️ Model not trained")
        st.caption("Run: `cd backend && python train.py`")

    st.markdown("---")
    st.caption("VGG16 + LSTM · Flickr8k")


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------
def render_home():
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📷 Image Caption Generator</h1>
            <p>AI-powered natural language descriptions using Deep Learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="card"><h3>🧠 Deep Learning</h3>'
            "<p>VGG16 CNN extracts visual features; LSTM generates fluent captions word-by-word.</p></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="card"><h3>⚡ Real-time Inference</h3>'
            "<p>Upload any JPG/PNG image and get an AI-generated description in seconds.</p></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="card"><h3>📊 Full Pipeline</h3>'
            "<p>Preprocessing, training, evaluation (BLEU), and a production-ready web interface.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### 🔄 How It Works")
    st.markdown(
        """
        ```
        Upload Image  →  Preprocess (224×224)  →  VGG16 Feature Extraction (4096-dim)
              ↓
        LSTM Decoder  →  Word-by-word Generation  →  Natural Language Caption
        ```
        """
    )

    info = get_model_info()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", info["dataset_name"])
    c2.metric("Images", info["num_images"] or "N/A")
    c3.metric("CNN", "VGG16")
    c4.metric("NLP", "LSTM")

    if st.button("🚀 Get Started — Upload an Image", use_container_width=True):
        st.session_state.page = "Upload"
        st.rerun()


# ---------------------------------------------------------------------------
# Page: Upload & Generate
# ---------------------------------------------------------------------------
def render_upload():
    st.markdown("## 🖼️ Upload & Generate Caption")

    col_img, col_cap = st.columns(2, gap="large")

    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag and drop or browse",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported: JPG, PNG, WEBP (max 10 MB)",
        )

        if uploaded:
            error = validate_upload(uploaded.name, uploaded.size)
            if error:
                st.error(error)
                uploaded = None

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.session_state.uploaded_image = image
            st.session_state.uploaded_filename = uploaded.name
            st.image(image, use_container_width=True, caption=f"{image.size[0]}×{image.size[1]} px")
            st.caption(f"📁 {uploaded.name}")

            if st.button("🗑️ Clear Upload", use_container_width=True):
                st.session_state.uploaded_image = None
                st.session_state.uploaded_filename = None
                st.session_state.caption_result = None
                st.rerun()
        else:
            st.info("👆 Upload a JPG or PNG image to begin.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cap:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ✨ Generated Caption")

        image = st.session_state.uploaded_image
        filename = st.session_state.uploaded_filename or "upload.jpg"
        if image and st.button("🚀 Generate Caption", use_container_width=True, type="primary"):
            if not is_model_available() and not st.session_state.use_api:
                st.error(
                    "Model not found. Train first:\n\n"
                    "```\ncd backend\npython train.py\n```"
                )
            else:
                with st.spinner("🔄 Extracting features & generating caption…"):
                    try:
                        if st.session_state.use_api:
                            buf = image_to_bytes(image)
                            result = generate_caption_api(buf, filename, api_url)
                        else:
                            result = generate_caption_direct(image)
                            if not st.session_state.use_api:
                                save_prediction(
                                    filename,
                                    result["caption"],
                                    result.get("confidence"),
                                )
                        st.session_state.caption_result = result
                        st.success("Caption generated successfully!")
                    except FileNotFoundError:
                        st.error("Model files missing. Run `cd backend && python train.py` first.")
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")

        result = st.session_state.caption_result
        if result and result.get("caption"):
            caption = result["caption"]
            st.markdown(
                f'<div class="caption-result">"{caption}"</div>',
                unsafe_allow_html=True,
            )

            conf = result.get("confidence")
            if conf is not None:
                st.progress(min(float(conf), 1.0))
                st.caption(f"Model confidence: **{conf:.1%}**")

            dl_col, regen_col = st.columns(2)
            with dl_col:
                st.download_button(
                    "⬇️ Download Caption",
                    data=caption,
                    file_name="caption.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with regen_col:
                if st.button("🔄 Generate Again", use_container_width=True):
                    st.session_state.caption_result = None
                    st.rerun()
        elif not image:
            st.markdown(
                "_Upload an image on the left, then click **Generate Caption**._"
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Prediction History
# ---------------------------------------------------------------------------
def render_history():
    st.markdown("## 📜 Prediction History")

    records = get_history(limit=50)

    if st.button("🗑️ Clear All History", type="secondary"):
        count = clear_all_history()
        st.success(f"Cleared {count} records.")
        st.rerun()

    if not records:
        st.info("No predictions yet. Generate a caption to see history here.")
        return

    for record in records:
        with st.expander(
            f"#{record['id']} — {record.get('image_name', 'Unknown')[:40]} "
            f"({record.get('date', '')} {record.get('time', '')})"
        ):
            st.write(f"**Caption:** {record.get('caption', '')}")
            if record.get("confidence"):
                st.caption(f"Confidence: {record['confidence']:.1%}")
            if st.button("Delete", key=f"del_{record['id']}"):
                delete_history_record(record["id"])
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Model Information
# ---------------------------------------------------------------------------
def render_model_info():
    st.markdown("## 🧠 Model Information")
    info = get_model_info()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="value">{info["dataset_name"]}</div>'
            f'<div class="label">Dataset</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="value">{info["num_images"] or "—"}</div>'
            f'<div class="label">Images</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        status = "✅ Trained" if info["model_trained"] else "❌ Not Trained"
        st.markdown(
            f'<div class="metric-card"><div class="value">{status}</div>'
            f'<div class="label">Model Status</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Architecture")
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown(
            '<div class="card"><h3>CNN Encoder</h3>'
            f"<p><b>{info['cnn_model']}</b></p>"
            "<p>Extracts 4096-dimensional feature vectors from 224×224 images.</p></div>",
            unsafe_allow_html=True,
        )
    with arch_col2:
        st.markdown(
            '<div class="card"><h3>NLP Decoder</h3>'
            f"<p><b>{info['nlp_model']}</b></p>"
            f"<p>Embedding: {info['embedding_dim']} | LSTM units: {info['lstm_units']}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Training Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train Accuracy", f"{info['accuracy']:.2%}" if info["accuracy"] else "N/A")
    m2.metric("Val Accuracy", f"{info['val_accuracy']:.2%}" if info["val_accuracy"] else "N/A")
    m3.metric("Train Loss", info["train_loss"] if info["train_loss"] else "N/A")
    m4.metric("Val Loss", info["val_loss"] if info["val_loss"] else "N/A")

    st.markdown("### BLEU Scores")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("BLEU-1", info["bleu1"] if info["bleu1"] else "N/A")
    b2.metric("BLEU-2", info["bleu2"] if info["bleu2"] else "N/A")
    b3.metric("BLEU-3", info["bleu3"] if info["bleu3"] else "N/A")
    b4.metric("BLEU-4", info["bleu4"] if info["bleu4"] else "N/A")

    # Training plots
    from config import ACCURACY_PLOT_PATH, LOSS_PLOT_PATH
    plot_col1, plot_col2 = st.columns(2)
    with plot_col1:
        if LOSS_PLOT_PATH.exists():
            st.image(str(LOSS_PLOT_PATH), caption="Training Loss", use_container_width=True)
    with plot_col2:
        if ACCURACY_PLOT_PATH.exists():
            st.image(str(ACCURACY_PLOT_PATH), caption="Training Accuracy", use_container_width=True)


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def render_about():
    st.markdown("## ℹ️ About Project")

    st.markdown(
        """
        <div class="about-section">
        <div class="card">
        <h3>Problem Statement</h3>
        <p>Images are universally understood but remain inaccessible to visually impaired users
        and difficult to index at scale. Manual annotation is expensive. This project automates
        image captioning using deep learning — combining computer vision and NLP.</p>

        <h3>Objectives</h3>
        <ul>
            <li>Implement VGG16-based feature extraction with transfer learning</li>
            <li>Build an LSTM decoder for sequence-to-sequence caption generation</li>
            <li>Preprocess the Flickr8k dataset with tokenization and vocabulary building</li>
            <li>Train, evaluate (BLEU), and deploy via Streamlit + FastAPI</li>
        </ul>

        <h3>Methodology</h3>
        <p>Pre-trained VGG16 extracts frozen 4096-dim features. An LSTM network learns to
        generate captions using teacher forcing during training and greedy decoding at inference.
        Data is split 80/10/10 for train/validation/test.</p>

        <h3>Applications</h3>
        <ul>
            <li>Accessibility tools for visually impaired users</li>
            <li>Social media auto-tagging and content management</li>
            <li>Image search engines and digital asset libraries</li>
            <li>E-commerce product description generation</li>
        </ul>

        <h3>Advantages</h3>
        <ul>
            <li>Transfer learning eliminates need to train CNN from scratch</li>
            <li>Feature caching speeds up training restarts</li>
            <li>Modular, production-ready architecture with SQLite history</li>
            <li>BLEU evaluation for objective quality measurement</li>
        </ul>

        <h3>Future Scope</h3>
        <ul>
            <li>Attention mechanisms (Bahdanau/Luong)</li>
            <li>Transformer-based decoders</li>
            <li>Training on MS COCO for better generalization</li>
            <li>Multi-language caption support</li>
            <li>Mobile deployment with model quantization</li>
        </ul>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 👨‍💻 Developer Information")
    st.info(
        "**Project:** Image Caption Generator Using Deep Learning\n\n"
        "**Stack:** Python · TensorFlow/Keras · VGG16 · LSTM · Streamlit · FastAPI · SQLite\n\n"
        "**Dataset:** Flickr8k (8,091 images, 5 captions each)"
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
page = st.session_state.page

if page == "Home":
    render_home()
elif page == "Upload":
    render_upload()
elif page == "History":
    render_history()
elif page == "Model Info":
    render_model_info()
elif page == "About":
    render_about()

st.markdown(
    '<p class="footer-text">Image Caption Generator Using Deep Learning · Internship Final Project</p>',
    unsafe_allow_html=True,
)
