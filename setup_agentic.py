# 🔥 FIXED SETUP SCRIPT
# RUN: python setup_agentic_fixed.py

import os
import sys

print("\n" + "="*60)
print("🚀 SETTING UP AGENTIC FEATURES")
print("="*60 + "\n")

# Check if we're in the right directory
if not os.path.exists('data'):
    print("❌ Error: Run this from your project root (where data/ folder is)")
    sys.exit(1)

print("Step 1: Checking project structure...")

# Create templates folder if missing
if not os.path.exists('templates'):
    os.makedirs('templates')
    print("✅ Created templates/ folder")

# Files to check
required_files = {
    'agent_models.py': 'Project root',
    'learning_agent.py': 'Project root',
    'templates/planner.html': 'templates folder'
}

missing = []
for file, location in required_files.items():
    if not os.path.exists(file):
        missing.append(f"{file} (should be in {location})")

if missing:
    print("\n❌ MISSING AGENTIC FILES:")
    for m in missing:
        print(f"   • {m}")
    print("\n📥 YOU NEED TO DOWNLOAD AND COPY:")
    print("   1. agent_models.py → project root")
    print("   2. learning_agent.py → project root")
    print("   3. planner.html → templates/ folder")
    print("\n   From the files I shared in chat!")
    sys.exit(1)

print("✅ All required files present!")

print("\nStep 2: Checking app.py imports...")

# FIXED: Use UTF-8 encoding
try:
    with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
        app_content = f.read()
except Exception as e:
    print(f"❌ Cannot read app.py: {e}")
    sys.exit(1)
    
if 'from agent_models import' not in app_content:
    print("❌ Your app.py doesn't import agent modules!")
    print("   Replace your app.py with app_FINAL.py from outputs folder")
    sys.exit(1)

print("✅ app.py has correct imports!")

print("\nStep 3: Testing imports...")

try:
    from agent_models import UserProfile, LearningGoal
    print("✅ agent_models.py imports successfully!")
except ImportError as e:
    print(f"❌ Cannot import agent_models: {e}")
    print("\n🔧 FIX:")
    print("   Make sure agent_models.py is in the same folder as app.py")
    sys.exit(1)

try:
    from learning_agent import LearningAgent
    print("✅ learning_agent.py imports successfully!")
except ImportError as e:
    print(f"❌ Cannot import learning_agent: {e}")
    print("\n🔧 FIX:")
    print("   Make sure learning_agent.py is in the same folder as app.py")
    sys.exit(1)

print("\n" + "="*60)
print("🎉 SUCCESS! AGENTIC MODE IS READY!")
print("="*60)
print("\nNow run: python app.py")
print("\nYou should see:")
print("✅ Backend ready!")
print("✅ Agentic mode ready!  ← THIS LINE!")
print("\nThen open: http://127.0.0.1:5000")
print("You'll see TWO buttons: [Quick Search] [Smart Planner]")
print("\n" + "="*60 + "\n")