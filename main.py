"""
AI STUDY ASSISTANT - PRODUCTION VERSION (LIGHTWEIGHT, NO TORCH)
Uses TF-IDF instead of sentence-transformers — same accuracy for keyword-driven
search, ~10x less RAM, works reliably on Render's free tier (512MB).
"""

import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings

warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================
CACHE_DIR = "cache"
COURSE_TFIDF_CACHE = f"{CACHE_DIR}/course_tfidf.pkl"       # (vectorizer, matrix)
PDF_TFIDF_CACHE = f"{CACHE_DIR}/pdf_tfidf.pkl"              # (vectorizer, matrix)
PDF_DATA_CACHE = f"{CACHE_DIR}/pdf_data.pkl"

# Global variables
df = None
course_vectorizer = None
X = None                  # course TF-IDF matrix
pdf_df = None
pdf_vectorizer = None
pdf_vectors = None        # pdf TF-IDF matrix
lemmatizer = None
stop_words = None


def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def save_to_cache(data, filename):
    ensure_cache_dir()
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_cache(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None


# ========================================
# INITIALIZATION
# ========================================
def initialize():
    """Ultra-fast initialization — no model to load anymore"""
    global df, lemmatizer, stop_words

    if df is not None:
        return

    print("🚀 Initializing AI Study Assistant...")

    # NLTK
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
    except Exception:
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

    # Load CSV
    print("📊 Loading course dataset...")
    df = pd.read_csv("data/udemy_courses.csv")
    df.fillna("", inplace=True)

    if "url" not in df.columns:
        df["url"] = ""

    # Combine fields for better matching
    df["combined_text"] = (
        df["course_title"].fillna("") + " " +
        df["subject"].fillna("") + " " +
        df["level"].fillna("") + " " +
        df["course_title"].fillna("")
    )

    df["processed_text"] = df["combined_text"].apply(clean_text)

    print(f"✅ Loaded {len(df)} courses")
    print("💡 TF-IDF index builds on first search — no heavy model to load")


def clean_text(text):
    """Enhanced text cleaning"""
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())

    if lemmatizer and stop_words:
        words = []
        for word in text.split():
            if word not in stop_words or len(word) > 3:
                words.append(lemmatizer.lemmatize(word))
        return " ".join(words)

    return text


# ========================================
# LAZY LOADING — TF-IDF INDEXES (fast, lightweight)
# ========================================
def ensure_embeddings_loaded():
    """Build (or load cached) TF-IDF index for courses. Same name kept
    so nothing else in the app needs to change."""
    global course_vectorizer, X

    if X is not None:
        return X

    cached = load_from_cache(COURSE_TFIDF_CACHE)
    if cached is not None:
        course_vectorizer, X = cached
        print("⚡ Loaded course TF-IDF index from cache")
        return X

    print("⏳ Building course TF-IDF index (first time, a few seconds)...")
    course_vectorizer = TfidfVectorizer(max_features=6000)
    X = course_vectorizer.fit_transform(df["processed_text"])
    save_to_cache((course_vectorizer, X), COURSE_TFIDF_CACHE)
    print("✅ Course index cached!")

    return X


def ensure_pdfs_loaded():
    """Build (or load cached) TF-IDF index for PDF filenames."""
    global pdf_df, pdf_vectorizer, pdf_vectors

    if pdf_df is not None:
        return pdf_df, pdf_vectors

    print("📄 Loading PDF list...")

    cached_df = load_from_cache(PDF_DATA_CACHE)
    cached_tfidf = load_from_cache(PDF_TFIDF_CACHE)

    if cached_df is not None and cached_tfidf is not None:
        pdf_df = cached_df
        pdf_vectorizer, pdf_vectors = cached_tfidf
        print(f"⚡ Loaded {len(pdf_df)} PDFs from cache")
        return pdf_df, pdf_vectors

    pdf_path = "data/Pdf"
    if not os.path.exists(pdf_path):
        pdf_df = pd.DataFrame(columns=["pdf_file", "clean_text"])
        pdf_vectors = None
        return pdf_df, pdf_vectors

    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith(".pdf")]
    if len(pdf_files) == 0:
        pdf_df = pd.DataFrame(columns=["pdf_file", "clean_text"])
        pdf_vectors = None
        return pdf_df, pdf_vectors

    print(f"   Found {len(pdf_files)} PDFs")
    pdf_data = []

    for pdf_file in pdf_files:
        filename_text = pdf_file.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        pdf_data.append({
            "pdf_file": pdf_file,
            "clean_text": clean_text(filename_text)
        })

    pdf_df = pd.DataFrame(pdf_data)

    pdf_vectorizer = TfidfVectorizer(max_features=3000)
    pdf_vectors = pdf_vectorizer.fit_transform(pdf_df["clean_text"])

    save_to_cache(pdf_df, PDF_DATA_CACHE)
    save_to_cache((pdf_vectorizer, pdf_vectors), PDF_TFIDF_CACHE)
    print(f"✅ Indexed {len(pdf_df)} PDFs by filename")

    return pdf_df, pdf_vectors


# ========================================
# RECOMMENDATION FUNCTIONS
# ========================================
def recommend_courses(course_title, top_n=5):
    """Recommend courses using TF-IDF cosine similarity"""
    ensure_embeddings_loaded()

    course_title_clean = clean_text(course_title)
    if not course_title_clean:
        return []

    query_vec = course_vectorizer.transform([course_title_clean])
    cosine_sim = cosine_similarity(query_vec, X).flatten()

    # TF-IDF similarity scores run lower than embedding similarity —
    # threshold tuned accordingly.
    similarity_threshold = 0.08
    top_indices = cosine_sim.argsort()[-top_n * 2:][::-1]

    filtered_results = []
    for idx in top_indices:
        if cosine_sim[idx] >= similarity_threshold:
            filtered_results.append(idx)
        if len(filtered_results) >= top_n:
            break

    if len(filtered_results) < 3:
        filtered_results = top_indices[:top_n]

    recommendations = df.iloc[filtered_results][["course_title", "url", "subject", "level"]]

    return [
        {
            "course_title": row["course_title"],
            "url": row["url"],
            "subject": row["subject"],
            "level": row["level"]
        }
        for _, row in recommendations.iterrows()
    ]


def recommend_pdfs(query, top_n=3):
    """Recommend PDFs based on filename matching (TF-IDF)"""
    ensure_pdfs_loaded()

    if pdf_df is None or len(pdf_df) == 0 or pdf_vectors is None:
        return []

    query_clean = clean_text(query)
    if not query_clean:
        return []

    query_vec = pdf_vectorizer.transform([query_clean])
    cosine_sim = cosine_similarity(query_vec, pdf_vectors).flatten()

    similarity_threshold = 0.06
    top_indices = cosine_sim.argsort()[-top_n * 2:][::-1]

    filtered_results = []
    seen_files = set()

    for idx in top_indices:
        pdf_file = pdf_df.iloc[idx]["pdf_file"]
        if cosine_sim[idx] >= similarity_threshold and pdf_file not in seen_files:
            filtered_results.append(idx)
            seen_files.add(pdf_file)
        if len(filtered_results) >= top_n:
            break

    if len(filtered_results) == 0 and len(top_indices) > 0:
        filtered_results = [top_indices[0]]

    pdf_recommendations = pdf_df.iloc[filtered_results]["pdf_file"]

    return [
        {
            "pdf_file": file,
            "url": f"/data_pdfs/{file}"
        }
        for file in pdf_recommendations
    ]


# ========================================
# PRODUCTION HELPERS
# ========================================
def get_stats():
    """Get system statistics"""
    return {
        "num_courses": len(df) if df is not None else 0,
        "num_pdfs": len(pdf_df) if pdf_df is not None else 0,
        "tfidf_ready": X is not None,
        "embeddings_cached": os.path.exists(COURSE_TFIDF_CACHE)
    }


if __name__ == "__main__":
    print("Run: python app.py")
