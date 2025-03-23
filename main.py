import pandas as pd
import re
import nltk
import os
import fitz  # PyMuPDF for extracting PDF text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
df = pd.read_csv("data/udemy_courses.csv")
df.fillna("", inplace=True)

# Ensure "url" column exists
if "url" not in df.columns:
    df["url"] = ""

# Combine Text Features
df["combined_text"] = df["course_title"] + " " + df["level"] + " " + df["subject"]

# Preprocessing Setup
nltk.download("stopwords")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Clean text by removing punctuation, stopwords, and lemmatizing"""
    text = re.sub(r"[^\w\s]", "", text.lower())
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# Apply Cleaning
df["processed_text"] = df["combined_text"].map(clean_text)

# 🔹 Lazy Load BERT Model
def get_model():
    global model
    if "model" not in globals():
        print("Loading BERT model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")  # ✅ Load before PDF extraction
    return model

# Load BERT model at the start
model = get_model()  # ✅ Load early to avoid delays later

# Convert Courses to Sentence Embeddings
X = model.encode(df["processed_text"].tolist(), convert_to_numpy=True)

# 🔹 Lazy Load PDFs
def extract_text_from_pdfs():
    global pdf_data
    if "pdf_data" not in globals():
        print("Loading PDFs...")
        pdf_texts = {}
        pdf_path = "data/Pdf"

        if not os.path.exists(pdf_path):
            print("❌ Error: PDF folder not found!")
            return pd.DataFrame(columns=["pdf_file", "text"])

        pdf_files = os.listdir(pdf_path)
        print(f"Found {len(pdf_files)} PDFs")

        for pdf_file in pdf_files:
            if pdf_file.endswith(".pdf"):
                try:
                    doc = fitz.open(f"{pdf_path}/{pdf_file}")
                    text = " ".join([doc[0].get_text()])  # ✅ Extract only the first page for now
                    pdf_texts[pdf_file] = text
                    print(f"✅ Extracted text from {pdf_file}")
                except Exception as e:
                    print(f"❌ Error loading {pdf_file}: {e}")

        pdf_data = pd.DataFrame(pdf_texts.items(), columns=["pdf_file", "text"])
    return pdf_data

# Load PDF Notes
pdf_data = extract_text_from_pdfs()

# Convert PDF Texts to Embeddings
pdf_df = pd.DataFrame(pdf_data, columns=["pdf_file", "text"])
pdf_vectors = model.encode(pdf_df["text"].tolist(), convert_to_numpy=True, batch_size=4)

# 🔹 Course Recommendation Function
def recommend_courses(course_title):
    print(f"🔍 Searching for: {course_title}")

    # Clean Input Text
    course_title_clean = clean_text(course_title)

    # Convert Input to BERT Embedding
    input_embedding = model.encode([course_title_clean], convert_to_numpy=True)

    # Compute Cosine Similarity
    cosine_sim = cosine_similarity(input_embedding, X).flatten()

    # Get Top 5 Most Similar Courses
    similar_indices = cosine_sim.argsort()[-4:][::-1]
    recommendations = df.iloc[similar_indices][["course_title", "url"]]

    formatted_recommendations = [
        {"course_title": row["course_title"], "url": row["url"]}
        for _, row in recommendations.iterrows()
    ]

    return formatted_recommendations  # ✅ Fixed Output Format

# 🔹 PDF Notes Recommendation Function
def recommend_pdfs(query):
    query_clean = clean_text(query)
    input_embedding = model.encode([query_clean], convert_to_numpy=True)

    # Compute Similarities for PDFs
    cosine_sim = cosine_similarity(input_embedding, pdf_vectors).flatten()
    similar_indices = cosine_sim.argsort()[-3:][::-1]
    pdf_recommendations = pdf_df.iloc[similar_indices]["pdf_file"]

    # ✅ Generate correct URLs (NO dictionary inside URL)
    formatted_pdfs = [{"pdf_file": file, "url": f"/data_pdfs/{file}"} for file in pdf_recommendations]

    return formatted_pdfs


