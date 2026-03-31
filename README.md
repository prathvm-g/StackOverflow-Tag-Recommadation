# Stack Overflow Tag Recommendation System

A multi-label Natural Language Processing (NLP) system that predicts relevant tags for Stack Overflow questions using two approaches: TF-IDF text features and Sentence-BERT semantic embeddings — both paired with a One-vs-Rest Linear SVM classifier, evaluated with ranking-based metrics including Micro-F1, Precision@5, and Hamming Loss.

---

## Problem Statement

Stack Overflow questions often receive incomplete or incorrect tags, making it difficult for the right experts to find and answer them. This project builds an automated tag recommendation system that suggests the most relevant tags based on the question title and body text.

---

## Dataset

| Property | Value |
|---|---|
| Source | Google BigQuery Public StackOverflow Dataset |
| Records | ~60,000 questions |
| Fields used | Question Title, Question Body, Tags |
| Label space | Top 50 most frequent tags |
| Format | CSV → Parquet (pipeline output) |

---

## Repository Structure
```
├── Data/
│   └── Stackoverflow_filtered.csv       # Raw input dataset
├── Models/                              # Saved model artefacts (.pkl files)
├── Notebook/
│   ├── 1-_Loading_data.ipynb            # Stage 1 — Data loading & cleaning
│   ├── 2-_Model_ready.ipynb             # Stage 2 — NLP preprocessing & tag filtering
│   ├── 3-_Modeling.ipynb                # Stage 3 — Model training (TF-IDF + SBERT)
│   └── 4-_Evaluation.ipynb             # Stage 4 — Evaluation & comparison
├── SQl Query/                           # BigQuery extraction queries
├── app.py                               # Streamlit inference app
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Pipeline

The project is structured as a 4-stage sequential pipeline. Each notebook takes a parquet file as input and produces one as output.
```
Stackoverflow_filtered.csv
        │
        ▼
┌─────────────────────────┐
│  1-_Loading_data.ipynb  │  Load CSV → handle nulls → parse tags → combine title+body
└────────────┬────────────┘
             │  Loading_data.parquet
             ▼
┌─────────────────────────┐
│  2-_Model_ready.ipynb   │  Clean HTML/URLs → select top-50 tags → filter rows
└────────────┬────────────┘
             │  model_ready.parquet
             ▼
┌─────────────────────────┐
│  3-_Modeling.ipynb      │  TF-IDF vectorization → train LinearSVC
└────────────┬────────────┘  SBERT embeddings (GPU) → train LinearSVC
             │  modeling.parquet + .pkl artifacts
             ▼
┌─────────────────────────┐
│  4-_Evaluation.ipynb    │  Micro-F1 · Precision@5 · Hamming Loss · model comparison
└─────────────────────────┘
```

### Stage 1 — Loading & Cleaning (`1-_Loading_data.ipynb`)
- Loads `Stackoverflow_filtered.csv`
- Fills missing values in title, body, and tag columns
- Parses pipe-delimited tag strings into Python lists
- Creates a combined `text` column (title + body)
- Outputs `Loading_data.parquet` with columns: `id`, `text`, `tag_list`

### Stage 2 — Model Ready (`2-_Model_ready.ipynb`)
- Cleans text: strips HTML tags (BeautifulSoup), removes URLs, and lowercases
- Counts tag frequencies across the corpus using `Counter`
- Selects the top 50 most frequent tags as the label space
- Filters each record to retain only those tags; removes records with no remaining tags
- Outputs `model_ready.parquet` with columns: `id`, `clean_text`, `filtered_tags`

### Stage 3 — Modelling (`3-_Modeling.ipynb`)
- Encodes labels with `MultiLabelBinarizer` → binary matrix `Y` of shape `(n, 50)`
- Builds TF-IDF features and trains a OneVsRest LinearSVC baseline
- Generates SBERT embeddings using `all-MiniLM-L6-v2` (GPU-accelerated) and trains a second LinearSVC on them
- Both models use identical `random_state=42 / test_size=0.2` splits for valid comparison
- Saves all sklearn artefacts via joblib; `SentenceTransformer` is **not** joblib-saved (see note below)

### Stage 4 — Evaluation (`4-_Evaluation.ipynb`)
- Loads all trained artefacts; reloads SBERT encoder directly from HuggingFace
- Computes Micro-F1, Precision@5, and Hamming Loss for both models
- Uses a combined ranking signal (`decision_function` score + normalised rank weight) for top-K selection
- Measures real inference latency per sample for both models
- Produces a side-by-side comparison table with actual computed values

---

## Model Configuration

### TF-IDF Vectorizer
| Parameter | Value |
|---|---|
| `max_features` | 50,000 |
| `ngram_range` | (1, 2) — unigrams + bigrams |
| `min_df` | 2 |
| `stop_words` | english |

### Classifier (both models)
| Parameter | Value |
|---|---|
| Type | `OneVsRestClassifier(LinearSVC)` |
| `max_iter` | 2,000 |
| `n_jobs` | -1 (all CPU cores) |

### Sentence-BERT
| Parameter | Value |
|---|---|
| Model ID | `all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| `batch_size` | 64 (GPU) / reduce to 32 on low VRAM |
| Device | CUDA — auto-detected, falls back to CPU |

---

## Model Performance

| Model | Micro F1 | Precision@5 | Hamming Loss |
|---|---|---|---|
| TF-IDF + LinearSVC | 0.68 | 0.25 | 0.014 |
| SBERT + LinearSVC (GPU) | 0.74 | 0.26 | 0.013 |

---

## Saved Artefacts

| File | Description |
|---|---|
| `mlb_encoder_tfidf.pkl` | `MultiLabelBinarizer` fitted on top-50 tags |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer (50k features, bigrams) |
| `tag_prediction_model_tfidf.pkl` | TF-IDF + LinearSVC classifier |
| `tag_prediction_model_sbert.pkl` | SBERT + LinearSVC classifier |

> **Why is there no `sbert_encoder.pkl`?**  
> `SentenceTransformer` wraps PyTorch modules. Joblib-pickling a GPU model does not reliably restore device state on reload. The encoder is always reloaded from the HuggingFace model ID (`all-MiniLM-L6-v2`) in both the evaluation notebook and the Streamlit app — this is guaranteed to be correct regardless of environment.

---

## Demo Application

A Streamlit web app provides an interactive interface for the tag recommendation system. Users input any question and receive the top 5 predicted tags from the trained TF-IDF model.
```bash
pip install -r requirements.txt
streamlit run app.py
```
Application link: https://stackoverflow-tag-recommandation.streamlit.app/
---

## Key Design Decisions

**Ranking over thresholding** — `decision_function` scores rank tags rather than applying a binary threshold. LinearSVC does not produce calibrated probabilities, so thresholding is unreliable.

**Combined ranking signal** — margin score is boosted by a normalised rank weight (`score + 0.25 × rank`) to surface high-confidence predictions over borderline ones.

**Identical train-test splits** — both models use `random_state=42` and `test_size=0.2` throughout the pipeline. Changing either value anywhere invalidates the comparison.

**GPU-accelerated SBERT** — device is auto-detected at runtime. `batch_size=64` is used on GPU; reduce to 32 if VRAM is limited.

**No joblib for SentenceTransformer** — only the downstream sklearn classifier is serialised. The encoder reloads from HuggingFace in seconds and is always in the correct state.

---

## Requirements
```
streamlit
scikit-learn
pandas
numpy
joblib
torch
sentence-transformers
beautifulsoup4
lxml
```

---

## Key Insights

- Multi-label problems require **ranking-based evaluation metrics** — accuracy is meaningless when multiple tags can be simultaneously correct.
- **TF-IDF has a structural advantage** for this task — programming tag names like `python`, `numpy`, and `flask` are literal tokens. Keyword matching is highly effective here; semantic embeddings do not add much when the signal is already in the surface form.
- **Feature representation impacts performance more than model selection** — both approaches use the same LinearSVC classifier. Any difference in metrics is entirely attributable to TF-IDF vs. SBERT features.
- **Label competition limits Precision@K** — the denominator is always 5, but many questions have 1–3 true tags. A perfect score is structurally unachievable.
- **GPU acceleration matters for SBERT** — embedding 60,000 documents on CPU is slow enough to be a bottleneck. On the GPU, the same operation runs in under a minute.
