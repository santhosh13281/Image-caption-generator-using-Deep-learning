"""Dark-theme CSS styles for the Streamlit frontend."""

DARK_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f1f5f9;
}

.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
}

.hero-banner h1 {
    margin: 0;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

.hero-banner p {
    margin: 0.75rem 0 0;
    font-size: 1.1rem;
    color: rgba(255,255,255,0.92);
}

.card {
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}

.card h3 {
    color: #a5b4fc;
    margin-top: 0;
    font-size: 1.1rem;
}

.caption-result {
    background: linear-gradient(135deg, rgba(102,126,234,0.2) 0%, rgba(118,75,162,0.2) 100%);
    border-left: 4px solid #667eea;
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    font-size: 1.15rem;
    font-style: italic;
    color: #f1f5f9;
    margin: 1rem 0;
    line-height: 1.6;
}

.metric-card {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
}

.metric-card .label {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}

.status-online { background: rgba(34,197,94,0.2); color: #4ade80; }
.status-offline { background: rgba(239,68,68,0.2); color: #f87171; }

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(102,126,234,0.5) !important;
}

div[data-testid="stFileUploader"] {
    background: rgba(30,41,59,0.5);
    border: 2px dashed rgba(129,140,248,0.4);
    border-radius: 12px;
    padding: 1rem;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.about-section h4 {
    color: #a5b4fc;
    margin-bottom: 0.5rem;
}

.about-section p, .about-section li {
    color: #cbd5e1;
    line-height: 1.7;
}

.footer-text {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    padding: 1rem 0;
}
</style>
"""
