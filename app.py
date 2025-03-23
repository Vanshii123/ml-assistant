import os
from flask import Flask, render_template, request, send_from_directory, url_for
from main import recommend_courses, recommend_pdfs

app = Flask(__name__)

# Serve images from the templates folder
@app.route('/templates/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'templates'), filename)

# ✅ FIXED: Single Home Route
@app.route("/", methods=["GET", "POST"])
def home():
    course_recommendations = []
    pdf_recommendations = []
    
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        print(f"Received query: {query}")  # Debugging

        if query:
            course_recommendations = recommend_courses(query)
            pdf_recommendations = recommend_pdfs(query)
            print(f"Course Recommendations: {course_recommendations}")  # Debugging
            print(f"PDF Recommendations: {pdf_recommendations}")  # Debugging
        else:
            print("No query provided.")

    return render_template("index.html", courses=course_recommendations, pdfs=pdf_recommendations)

# ✅ Route to Serve PDFs from `data/Pdf/`
@app.route("/data_pdfs/<path:filename>")
def serve_pdf(filename):
    pdf_directory = os.path.abspath("data/Pdf")  # Ensure absolute path
    file_path = os.path.join(pdf_directory, filename)

    if not os.path.exists(file_path):
        print(f"🚨 File Not Found: {file_path}")
        return "File not found", 404

    return send_from_directory(pdf_directory, filename)

print("🔥 Flask App Running & Rendering index.html")

if __name__ == "__main__":
    app.run(debug=True)





























































































































































