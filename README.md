# **📚 AI-Powered Personal Study Assistant**

🚀 **An intelligent ML-based assistant that recommends learning resources (YouTube videos & PDFs) based on user queries!**

## **✨ Features**

✅ **AI-driven personalized recommendations** for videos and PDFs
✅ **BERT-based similarity matching** for better accuracy
✅ **TF-IDF + Content-Based Recommendation**
✅ **Flask API for real-time queries**
✅ **User-friendly interface with a modern UI**


## **🛠️ Tech Stack**

* **Machine Learning:** BERT, TF-IDF, Scikit-Learn
* **Backend:** Python, Flask
* **Frontend:** HTML, CSS, JavaScript
* **Dataset:** Kaggle Udemy Courses & PDF Notes
* **Deployment:** Render (Backend) + Vercel (Frontend)


## **📂 Dataset**

The assistant is trained on **Kaggle datasets**:
1️⃣ **Udemy Courses Dataset** – Used to recommend videos
2️⃣ **PDF Notes Dataset** – Extracted from a collection of study materials

📌 **Preprocessing Includes:**
✔ Text cleaning (removing punctuation, stopwords)
✔ TF-IDF vectorization
✔ Sentence embeddings using **BERT (all-MiniLM-L6-v2)**

---

## **🚀 How It Works?**

1️⃣ **User enters a study topic** (e.g., "Machine Learning Basics")
2️⃣ **AI Assistant fetches relevant videos & PDFs**
3️⃣ **ML Model ranks results based on relevance**
4️⃣ **User clicks on the links to access resources**

---

## **🖥️ Setup & Installation**

### **🔹 1. Clone the Repository**

```sh
git clone https://github.com/Vanshii123/ml-assistant.git
cd ml-assistant
```

### **🔹 2. Install Dependencies**

```sh
pip install -r requirements.txt
```

### **🔹 3. Run the Flask App**

```sh
python app.py
```

🌐 Open **`http://127.0.0.1:5000/`** in your browser to use the assistant!


## **👩‍💻 Author**

💡 Developed by **Vanshika Chauhan**
📩 **Email:** (rv.chauhan322@gmail.com)
🔗 **LinkedIn:** (https://www.linkedin.com/vanshika-chauhan-1ba100279/)


## 📜 License

This project is licensed under the **MIT License** – feel free to use and improve it!


