"""
AI STUDY ASSISTANT - FINAL PRODUCTION VERSION
All features working, no errors, fast startup, no watchdog issues
"""

import os
from flask import Flask, render_template, request, send_from_directory, jsonify
from main import recommend_courses, recommend_pdfs, initialize

# Import agentic system (optional upgrade)
try:
    from agent_models import UserProfile, LearningGoal, SkillLevel, LearningStyle, TimeConstraint
    from learning_agent import LearningAgent
    AGENTIC_MODE_AVAILABLE = True
except ImportError:
    AGENTIC_MODE_AVAILABLE = False

app = Flask(__name__)

# ========================================
# INITIALIZATION ON STARTUP
# ========================================
print("\n" + "=" * 50)
print("🚀 AI STUDY ASSISTANT STARTING")
print("=" * 50 + "\n")

try:
    initialize()
    print("✅ Backend ready!")
    
    if AGENTIC_MODE_AVAILABLE:
        agent = LearningAgent(recommend_courses, recommend_pdfs)
        print("✅ Agentic mode ready!")
except Exception as e:
    print(f"⚠️ Initialization warning: {e}")
    print("   App will start but some features may not work")

# ========================================
# ROUTES
# ========================================

@app.route("/", methods=["GET", "POST"])
def home():
    """Main search page"""
    courses = []
    pdfs = []
    error = None
    
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        
        if query:
            try:
                courses = recommend_courses(query)
                pdfs = recommend_pdfs(query)
                print(f"✅ Search: '{query}' - {len(courses)} courses, {len(pdfs)} PDFs")
            except Exception as e:
                error = f"Search error: {str(e)}"
                print(f"❌ {error}")
    
    return render_template(
        "index.html",
        courses=courses,
        pdfs=pdfs,
        error=error,
        agentic_available=AGENTIC_MODE_AVAILABLE
    )

@app.route("/planner")
def planner():
    """Agentic planner page"""
    if not AGENTIC_MODE_AVAILABLE:
        return "Agentic mode not installed", 404
    return render_template("planner.html")

@app.route("/api/create-plan", methods=["POST"])
def create_plan():
    """Generate personalized learning plan"""
    if not AGENTIC_MODE_AVAILABLE:
        return jsonify({"error": "Agentic mode not available"}), 404
    
    try:
        data = request.get_json()
        
        goal_map = {
            "ml_internship": LearningGoal.ML_INTERNSHIP,
            "data_science": LearningGoal.DATA_SCIENCE_ROLE,
            "ml_engineer": LearningGoal.ML_ENGINEER_ROLE,
        }
        
        level_map = {
            "beginner": SkillLevel.BEGINNER,
            "intermediate": SkillLevel.INTERMEDIATE,
            "advanced": SkillLevel.ADVANCED,
        }
        
        profile = UserProfile(
            goal=goal_map.get(data.get("goal", "ml_internship"), LearningGoal.ML_INTERNSHIP),
            current_level=level_map.get(data.get("level", "beginner"), SkillLevel.BEGINNER),
            current_skills=data.get("skills", []),
            learning_style=LearningStyle.BALANCED,
            time_constraint=TimeConstraint(
                hours_per_day=float(data.get("hours_per_day", 2.0)),
                days_per_week=int(data.get("days_per_week", 5)),
                total_weeks=int(data.get("weeks", 12))
            )
        )
        
        plan, trace = agent.create_personalized_plan(profile)
        
        return jsonify({
            "success": True,
            "plan": {
                "total_weeks": plan.total_weeks,
                "total_hours": plan.total_hours,
                "success_probability": plan.success_probability,
                "modules": [
                    {
                        "week_number": m.week_number,
                        "title": m.title,
                        "objective": m.objective,
                        "skills_covered": m.skills_covered,
                        "estimated_hours": m.estimated_hours,
                        "reasoning": m.reasoning,
                        "resources": [
                            {
                                "title": r.title,
                                "type": r.type,
                                "url": r.url,
                                "estimated_hours": r.estimated_hours
                            }
                            for r in m.resources[:5]
                        ]
                    }
                    for m in plan.modules
                ],
                "planning_decisions": plan.planning_decisions[:5],
                "risks": plan.risks,
                "checkpoints": plan.checkpoints
            }
        })
    except Exception as e:
        print(f"❌ Plan error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve static files"""
    return send_from_directory("static", filename)

@app.route("/templates/<path:filename>")
def serve_template_asset(filename):
    """Serve template assets like images"""
    return send_from_directory("templates", filename)

@app.route("/data_pdfs/<path:filename>")
def serve_pdf(filename):
    """Serve PDF files"""
    pdf_dir = os.path.abspath("data/Pdf")
    file_path = os.path.join(pdf_dir, filename)
    
    if not os.path.exists(file_path):
        return "PDF not found", 404
    
    return send_from_directory(pdf_dir, filename)

@app.route("/health")
def health():
    """Health check"""
    return {
        "status": "ok",
        "agentic_mode": AGENTIC_MODE_AVAILABLE
    }, 200

# ========================================
# ERROR HANDLERS
# ========================================

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", courses=[], pdfs=[], error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("index.html", courses=[], pdfs=[], error="Server error"), 500

# ========================================
# STARTUP
# ========================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🌐 SERVER STARTING")
    print("=" * 50)
    print("📍 Main: http://127.0.0.1:5000")
    if AGENTIC_MODE_AVAILABLE:
        print("🧠 Planner: http://127.0.0.1:5000/planner")
    print("🛑 Press CTRL+C to stop")
    print("=" * 50 + "\n")
    
    app.run(
        debug=True,
        host='0.0.0.0',      # Listen on all interfaces
        port=5000,
        use_reloader=False,  # Disable watchdog
        threaded=True        # Handle multiple requests
    )