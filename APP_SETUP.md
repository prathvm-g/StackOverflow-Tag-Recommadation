# SO Tag Recommender — App Setup Guide

## Required Files

Before running the app, make sure these files are all in the **same folder** as `app.py`:

```
app.py
requirements.txt
packages.txt
.streamlit/
    config.toml
tag_prediction_model_tfidf.pkl     ← from 3-_Modeling.ipynb
tfidf_vectorizer.pkl               ← from 3-_Modeling.ipynb
mlb_encoder_tfidf.pkl              ← from 3-_Modeling.ipynb
```

The three `.pkl` files are produced by running `3-_Modeling.ipynb` end-to-end.
They are saved by the cell under "Save TF-IDF Model Artifacts".

---

## Run Locally

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

The app opens at http://localhost:8501

---

## Deploy to Streamlit Cloud

### Step 1 — Push to GitHub

Your repo must contain at minimum:

```
app.py
requirements.txt
packages.txt
.streamlit/config.toml
tag_prediction_model_tfidf.pkl
tfidf_vectorizer.pkl
mlb_encoder_tfidf.pkl
```

> If your .pkl files are large (>100MB), use Git LFS:
> ```bash
> git lfs install
> git lfs track "*.pkl"
> git add .gitattributes
> ```

### Step 2 — Connect to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **New app**
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy**

### Step 3 — Verify

Streamlit Cloud automatically:
- Installs system packages from `packages.txt` (needed for `lxml`)
- Installs Python packages from `requirements.txt`
- Uses `.streamlit/config.toml` for theme

---

## What was fixed from the original app.py

| Issue | Original | Fixed |
|---|---|---|
| `argsort` bug | `np.argsort(scores[0])[-top_k:]` returns ascending order — tags displayed lowest-confidence first | `np.argsort(combined)[::-1][:top_k]` — correctly descending |
| No preprocessing | Raw user input passed directly to vectorizer — train/serve skew | `clean_text()` applied: strips HTML, removes URLs, lowercases — matches training pipeline |
| No ranking signal | Raw scores only | Combined signal: `score + 0.25 × rank_weight` — matches evaluation notebook |
| Wrong artifact names | Loads `tag_prediction_model.pkl` / `mlb_encoder.pkl` | Loads `tag_prediction_model_tfidf.pkl` / `mlb_encoder_tfidf.pkl` — matches modeling notebook output |
| No caching | Models reloaded on every interaction | `@st.cache_resource` — loaded once, reused |
| No confidence display | Tags shown as plain list | Ranked chips + normalised confidence bars |
