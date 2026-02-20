import streamlit as st
import joblib
import numpy as np
import os

# -------------------------
# Load Model Artifacts
# -------------------------

MODEL_PATH = "tag_prediction_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
ENCODER_PATH = "mlb_encoder.pkl"
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    mlb = joblib.load(ENCODER_PATH)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Tag Recommender", layout="centered")

st.title("Stack Overflow Tag Recommendation")
st.write("Enter a question title or description to get recommended tags.")

user_input = st.text_area("Enter your question text:", height=150)

if st.button("Predict Tags"):

    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        text_vec = vectorizer.transform([user_input])
        scores = model.decision_function(text_vec)

        top_k = 5
        top_indices = np.argsort(scores[0])[-top_k:]
        tags = mlb.classes_[top_indices]

        st.success("Recommended Tags:")
        st.write(list(tags))