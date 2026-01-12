import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from auth import register_user, authenticate_user
from detector import detect_anomaly

# =====================================================
# APP SETUP
# =====================================================
app = Flask(__name__)

# ✅ Allow Netlify + Local
CORS(app)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =====================================================
# HELPERS
# =====================================================
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    return jsonify({"message": "Visual Anomaly Detection Backend Running"})

# ---------------- AUTH ----------------
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    success, message = register_user(
        data.get("username"),
        data.get("password")
    )
    return jsonify({"success": success, "message": message})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    success, role = authenticate_user(
        data.get("username"),
        data.get("password")
    )
    return jsonify({"success": success, "role": role})

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    username = request.form.get("username", "unknown")

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Run model
    result = detect_anomaly(image_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    BASE_URL = request.host_url.rstrip("/")

    history = load_history()
    history.append({
        "username": username,
        "image": f"{BASE_URL}/uploads/{filename}",
        "result": result["label"],
        "score": result["score"],
        "time": timestamp,
        "outline_image": f"{BASE_URL}/results/{os.path.basename(result['outline_image'])}",
        "filled_image": f"{BASE_URL}/results/{os.path.basename(result['filled_image'])}"
    })
    save_history(history)

    return jsonify({
        "result": result["label"],
        "anomaly_score": result["score"],
        "outline_image": f"{BASE_URL}/results/{os.path.basename(result['outline_image'])}",
        "filled_image": f"{BASE_URL}/results/{os.path.basename(result['filled_image'])}"
    })

# ---------------- FILE SERVING ----------------
@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route("/history/<username>")
def history(username):
    data = load_history()
    return jsonify([h for h in data if h["username"] == username])

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
