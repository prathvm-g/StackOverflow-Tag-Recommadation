import streamlit as st
import joblib
import numpy as np
import re
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SO Tag Recommender",
    page_icon="🏷️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 720px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f78166;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}
.hero .subtitle {
    font-size: 0.9rem;
    color: #7d8590;
    font-family: 'JetBrains Mono', monospace;
}
.hero .subtitle span {
    color: #3fb950;
}

/* ── Section labels ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
}

/* ── Text area override ── */
textarea {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    caret-color: #3fb950 !important;
}
textarea:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(56, 139, 253, 0.15) !important;
}

/* ── Predict button ── */
.stButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.8rem !important;
    letter-spacing: 0.5px !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Tag chips ── */
.tags-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 1.2rem;
    padding: 1.4rem;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
}
.tag-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1f2937;
    border: 1px solid #388bfd44;
    color: #79c0ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 6px;
    cursor: default;
    transition: border-color 0.15s;
}
.tag-chip:hover {
    border-color: #388bfd;
    background: #1c2e45;
}
.tag-rank {
    font-size: 0.68rem;
    color: #3fb950;
    font-weight: 700;
}

/* ── Confidence bar ── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 7px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.conf-label {
    width: 130px;
    color: #79c0ff;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex-shrink: 0;
}
.conf-bar-bg {
    flex: 1;
    height: 6px;
    background: #21262d;
    border-radius: 4px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #388bfd, #79c0ff);
}
.conf-score {
    width: 44px;
    text-align: right;
    color: #7d8590;
    font-size: 0.72rem;
}

/* ── Info / warning / error boxes ── */
.info-box {
    background: #0d2137;
    border: 1px solid #388bfd44;
    border-left: 3px solid #388bfd;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #7d8590;
    margin-top: 1rem;
    font-family: 'JetBrains Mono', monospace;
}
.warn-box {
    background: #1c1507;
    border-left: 3px solid #d29922;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #d29922;
    font-family: 'JetBrains Mono', monospace;
}
.err-box {
    background: #1a0a0a;
    border-left: 3px solid #f85149;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    color: #f85149;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Divider ── */
hr { border-color: #21262d !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #30363d;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #21262d;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model      = joblib.load("tag_prediction_model_tfidf.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    mlb        = joblib.load("mlb_encoder_tfidf.pkl")
    
    # Verify vectorizer is actually fitted
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(vectorizer, attributes=["idf_"])
    
    return model, vectorizer, mlb

try:
    model, vectorizer, mlb = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    load_error = str(e)


# ─────────────────────────────────────────────
# Text preprocessing — matches training pipeline
# (Stage 2: 2-_Model_ready.ipynb)
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Mirrors the clean_text() function used during training:
      - Strip HTML tags (BeautifulSoup + lxml)
      - Remove URLs
      - Lowercase
    Applying the same cleaning at inference prevents train-serve skew.
    """
    text = BeautifulSoup(text, "lxml").get_text()
    text = re.sub(r"http\S+", "", text)
    return text.lower().strip()


# ─────────────────────────────────────────────
# Prediction with combined ranking signal
# (matches 4-_Evaluation.ipynb logic)
# ─────────────────────────────────────────────
def predict_tags(raw_text: str, top_k: int = 5):
    """
    Returns a list of (tag, normalised_confidence) tuples, sorted
    highest-confidence first.

    Ranking approach:
      combined_score = decision_score + 0.25 × normalised_rank_weight
    This matches the evaluation notebook exactly — margin score is
    boosted by rank to surface high-confidence predictions.
    """
    # 1. Preprocess — same pipeline as training
    cleaned = clean_text(raw_text)

    # 2. Vectorize
    vec = vectorizer.transform([cleaned])

    # 3. Raw decision scores (one per class)
    scores = model.decision_function(vec)[0]   # shape: (n_classes,)

    # 4. Combined ranking signal
    order = np.argsort(scores)                 # ascending
    rank_weight = np.zeros_like(scores)
    rank_weight[order] = np.linspace(0, 1, len(scores))
    combined = scores + 0.25 * rank_weight

    # 5. Top-K — descending confidence order ([::-1] fix)
    top_indices = np.argsort(combined)[::-1][:top_k]
    top_scores  = combined[top_indices]

    # 6. Normalise scores to [0, 1] for display only
    s_min, s_max = top_scores.min(), top_scores.max()
    if s_max > s_min:
        norm_scores = (top_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones_like(top_scores)

    tags = mlb.classes_[top_indices]
    return list(zip(tags, norm_scores.tolist()))


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <h1>🏷️ SO Tag Recommender</h1>
    <div class="subtitle">
        TF-IDF + LinearSVC &nbsp;·&nbsp;
        <span>top-50 tags</span> &nbsp;·&nbsp;
        ranking-based selection
    </div>
</div>
""", unsafe_allow_html=True)

# Artifact load error
if not artifacts_ok:
    st.markdown(f"""
    <div class="err-box">
        ❌ &nbsp;Failed to load model artifacts.<br><br>
        <code>{load_error}</code><br><br>
        Make sure these files are in the same directory as app.py:<br>
        &nbsp;&nbsp;• tag_prediction_model_tfidf.pkl<br>
        &nbsp;&nbsp;• tfidf_vectorizer.pkl<br>
        &nbsp;&nbsp;• mlb_encoder_tfidf.pkl
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Input
st.markdown('<div class="section-label">Question Text</div>', unsafe_allow_html=True)
user_input = st.text_area(
    label="",
    placeholder="Paste your Stack Overflow question title and/or body here...",
    height=160,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("⚡  Predict Tags", use_container_width=True)

# Prediction
if predict_clicked:
    if not user_input.strip():
        st.markdown('<div class="warn-box">⚠️ &nbsp;Please enter some question text first.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Running inference..."):
            results = predict_tags(user_input, top_k=5)

        # Tag chips
        chip_html = '<div class="tags-container">'
        for rank, (tag, _) in enumerate(results, 1):
            chip_html += f'<div class="tag-chip"><span class="tag-rank">#{rank}</span>{tag}</div>'
        chip_html += "</div>"

        st.markdown('<div class="section-label" style="margin-top:1.8rem;">Recommended Tags</div>', unsafe_allow_html=True)
        st.markdown(chip_html, unsafe_allow_html=True)

        # Confidence bars
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Confidence</div>', unsafe_allow_html=True)
        bars_html = ""
        for tag, conf in results:
            pct = int(conf * 100)
            bars_html += f"""
            <div class="conf-row">
                <div class="conf-label">{tag}</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct}%"></div>
                </div>
                <div class="conf-score">{pct}%</div>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)

        # Info note
        st.markdown("""
        <div class="info-box">
            ℹ️ &nbsp;Scores are normalised for display. The model ranks all 50 tags
            by decision score + rank weight, then returns the top 5.
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Stack Overflow Tag Recommender &nbsp;·&nbsp; TF-IDF + LinearSVC &nbsp;·&nbsp; ~60k training questions
</div>
""", unsafe_allow_html=True)
