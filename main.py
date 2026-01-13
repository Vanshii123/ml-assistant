"""
AI STUDY ASSISTANT - PRODUCTION VERSION
Better error handling, improved relevancy, production-ready
"""

import pandas as pd
import re
import nltk
import os
import fitz
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings

# Suppress MuPDF warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================
CACHE_DIR = "cache"
COURSE_EMBEDDINGS_CACHE = f"{CACHE_DIR}/course_embeddings.pkl"
PDF_EMBEDDINGS_CACHE = f"{CACHE_DIR}/pdf_embeddings.pkl"

# Global variables
model = None
df = None
X = None
pdf_df = None
pdf_vectors = None
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
    """Ultra-fast initialization"""
    global df, lemmatizer, stop_words
    
    if df is not None:
        return
    
    print("🚀 Initializing AI Study Assistant...")
    
    # NLTK
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))
    except:
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
    
    # IMPROVED: Add more fields for better matching
    # Combine title, subject, level for richer context
    df["combined_text"] = (
        df["course_title"].fillna("") + " " + 
        df["subject"].fillna("") + " " + 
        df["level"].fillna("") + " " +
        df["course_title"].fillna("")  # Repeat title for emphasis
    )
    
    df["processed_text"] = df["combined_text"].apply(clean_text)
    
    print(f"✅ Loaded {len(df)} courses")
    print("💡 Model loads on first search")

def clean_text(text):
    """Enhanced text cleaning"""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Remove extra spaces
    text = " ".join(text.split())
    
    if lemmatizer and stop_words:
        # Lemmatize but keep important words
        words = []
        for word in text.split():
            if word not in stop_words or len(word) > 3:  # Keep longer stop words
                words.append(lemmatizer.lemmatize(word))
        return " ".join(words)
    
    return text

# ========================================
# LAZY LOADING
# ========================================
def ensure_model_loaded():
    """Load model on first use"""
    global model
    
    if model is None:
        print("\n⏳ Loading AI model (one-time, 10-15 seconds)...")
        model = SentenceTransformer("all-mpnet-base-v2")
        print("✅ Model ready!\n")
    
    return model

def ensure_embeddings_loaded():
    """Load course embeddings on first use"""
    global X
    
    if X is None:
        print("⏳ Loading course embeddings...")
        
        X = load_from_cache(COURSE_EMBEDDINGS_CACHE)
        
        if X is not None:
            print("⚡ Loaded from cache!")
            return X
        
        # First time - compute embeddings
        print("⏳ Computing embeddings (first time, ~1-2 min)...")
        ensure_model_loaded()
        
        batch_size = 500
        embeddings_list = []
        
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_texts = df["processed_text"].iloc[i:batch_end].tolist()
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings_list.append(batch_embeddings)
            print(f"   Batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
        
        X = np.vstack(embeddings_list)
        save_to_cache(X, COURSE_EMBEDDINGS_CACHE)
        print("✅ Embeddings cached!")
    
    return X

def ensure_pdfs_loaded():
    """Load PDFs with better error handling"""
    global pdf_df, pdf_vectors
    
    if pdf_df is not None:
        return pdf_df, pdf_vectors
    
    print("📄 Loading PDF notes...")
    
    # Try cache
    cached_df = load_from_cache(f"{CACHE_DIR}/pdf_data.pkl")
    cached_vectors = load_from_cache(PDF_EMBEDDINGS_CACHE)
    
    if cached_df is not None and cached_vectors is not None:
        pdf_df = cached_df
        pdf_vectors = cached_vectors
        print(f"⚡ Loaded {len(pdf_df)} PDFs from cache")
        return pdf_df, pdf_vectors
    
    # Process PDFs with better error handling
    pdf_path = "data/Pdf"
    if not os.path.exists(pdf_path):
        pdf_df = pd.DataFrame(columns=["pdf_file", "text", "clean_text"])
        pdf_vectors = np.array([])
        return pdf_df, pdf_vectors
    
    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith(".pdf")]
    if len(pdf_files) == 0:
        pdf_df = pd.DataFrame(columns=["pdf_file", "text", "clean_text"])
        pdf_vectors = np.array([])
        return pdf_df, pdf_vectors
    
    print(f"   Processing {len(pdf_files)} PDFs...")
    pdf_data = []
    
    for pdf_file in pdf_files:
        try:
            # IMPROVED: Extract ALL pages, not just first
            doc = fitz.open(f"{pdf_path}/{pdf_file}")
            
            # Extract text from all pages (max 10 pages for performance)
            pages_to_extract = min(len(doc), 10)
            full_text = ""
            
            for page_num in range(pages_to_extract):
                try:
                    page_text = doc[page_num].get_text()
                    full_text += page_text + " "
                except:
                    continue
            
            doc.close()
            
            if full_text.strip():
                # IMPROVED: Use filename as additional context
                filename_without_ext = pdf_file.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
                enhanced_text = filename_without_ext + " " + full_text
                
                pdf_data.append({
                    "pdf_file": pdf_file,
                    "text": full_text.strip(),
                    "clean_text": clean_text(enhanced_text)
                })
        
        except Exception as e:
            # Silently skip problematic PDFs (MuPDF errors)
            continue
    
    if not pdf_data:
        pdf_df = pd.DataFrame(columns=["pdf_file", "text", "clean_text"])
        pdf_vectors = np.array([])
        return pdf_df, pdf_vectors
    
    pdf_df = pd.DataFrame(pdf_data)
    
    # Encode using clean text
    ensure_model_loaded()
    pdf_vectors = model.encode(
        pdf_df["clean_text"].tolist(), 
        convert_to_numpy=True, 
        show_progress_bar=False
    )
    
    # Cache
    save_to_cache(pdf_df, f"{CACHE_DIR}/pdf_data.pkl")
    save_to_cache(pdf_vectors, PDF_EMBEDDINGS_CACHE)
    print(f"✅ Processed {len(pdf_df)} PDFs successfully")
    
    return pdf_df, pdf_vectors

# ========================================
# RECOMMENDATION FUNCTIONS
# ========================================
def recommend_courses(course_title, top_n=5):
    """
    Recommend courses with improved relevancy
    Returns top_n most similar courses
    """
    ensure_model_loaded()
    ensure_embeddings_loaded()
    
    # Clean and encode query
    course_title_clean = clean_text(course_title)
    if not course_title_clean:
        return []
    
    # Encode query
    input_embedding = model.encode([course_title_clean], convert_to_numpy=True)
    
    # Calculate similarity
    cosine_sim = cosine_similarity(input_embedding, X).flatten()
    
    # IMPROVED: Get more results and filter by minimum threshold
    similarity_threshold = 0.3  # Only show courses with >30% similarity
    
    # Get top candidates
    top_indices = cosine_sim.argsort()[-top_n*2:][::-1]  # Get 2x results
    
    # Filter by threshold
    filtered_results = []
    for idx in top_indices:
        if cosine_sim[idx] >= similarity_threshold:
            filtered_results.append(idx)
        if len(filtered_results) >= top_n:
            break
    
    # If too few results, lower threshold
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
    """
    Recommend PDFs with better relevancy
    Returns top_n most similar PDFs
    """
    ensure_model_loaded()
    ensure_pdfs_loaded()
    
    if pdf_df is None or len(pdf_df) == 0:
        return []
    
    # Clean query
    query_clean = clean_text(query)
    if not query_clean:
        return []
    
    # Encode query
    input_embedding = model.encode([query_clean], convert_to_numpy=True)
    
    # Calculate similarity
    cosine_sim = cosine_similarity(input_embedding, pdf_vectors).flatten()
    
    # IMPROVED: Filter by threshold
    similarity_threshold = 0.25
    
    # Get top results
    top_indices = cosine_sim.argsort()[-top_n*2:][::-1]
    
    # Filter by threshold and avoid duplicates
    filtered_results = []
    seen_files = set()
    
    for idx in top_indices:
        pdf_file = pdf_df.iloc[idx]["pdf_file"]
        if cosine_sim[idx] >= similarity_threshold and pdf_file not in seen_files:
            filtered_results.append(idx)
            seen_files.add(pdf_file)
        if len(filtered_results) >= top_n:
            break
    
    # If no results meet threshold, return top results anyway
    if len(filtered_results) == 0 and len(top_indices) > 0:
        filtered_results = [top_indices[0]]  # At least return the top match
    
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
        "model_loaded": model is not None,
        "embeddings_cached": os.path.exists(COURSE_EMBEDDINGS_CACHE)
    }

if __name__ == "__main__":
    print("Run: python app.py")

