# Stack Overflow Tag Recommendation System

A multi-label Natural Language Processing (NLP) system that predicts relevant tags for Stack Overflow questions using TF-IDF text features and a Linear Support Vector Machine (Linear SVM) classifier.

---

## Problem Statement

Stack Overflow questions often receive incomplete or incorrect tags, making it difficult for the right experts to find and answer them.  
This project builds an automated tag recommendation system that suggests the most relevant tags based on the question title and body text.

---

## Dataset

- Source: Google BigQuery Public StackOverflow Dataset  
- Records Used: ~60,000 questions  
- Fields Used:
  - Question Title
  - Question Body
  - Tags

The top 50 most frequent tags were selected to create a balanced multi-label classification space.

---

## Project Pipeline

1. Data extraction from BigQuery
2. Data preprocessing and text cleaning
3. Tag filtering (Top 50 tags)
4. Feature engineering using TF-IDF
5. Multi-label classification using One-vs-Rest Linear SVM
6. Ranking-based evaluation

---

## Model Configuration

**TF-IDF Vectorizer**
- max_features = 50,000  
- ngram_range = (1, 2)  
- min_df = 2  
- stop_words = "english"

**Classifier**
- OneVsRest LinearSVC

---

## Model Performance

| Metric | Score |
|-------|------|
| Micro F1 Score | 0.68 |
| Precision@5 | 0.25 |
| Hamming Loss | 0.014 |

---

## Demo application based on the model 

Streamlit-based web application that provides an interactive interface for the Stack Overflow Tag Recommendation System.  
Users can input a question and receive the top 5 predicted tags based on the trained Linear SVM model.

## Key Insights

- Multi-label problems require ranking-based evaluation metrics rather than simple classification accuracy.
- Feature engineering significantly impacts performance in NLP tag recommendation systems.
- Label competition between similar tags limits achievable Precision@K scores.
- Decision-score ranking enables effective tag recommendation without probability calibration.

---
