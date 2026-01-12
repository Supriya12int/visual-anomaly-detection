import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from auth import register_user, authenticate_user
from detector import detect_anomaly

app = Flask(__name__)

# ✅ Allow ALL Netlify origins (simple + safe)
CORS(app)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------------- HISTORY ----------------
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return jsonify({"message": "Backend running"})

# ---------- AUTH ----------
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    success, message = register_user(data["username"], data["password"])
    return jsonify({"success": success, "message": message})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    success, role = authenticate_user(data["username"], data["password"])
    return jsonify({"success": success, "role": role})

# ---------- PREDICT ----------
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

# ---------- FILE SERVING ----------
@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/results/<path:filename>")
def results(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route("/history/<username>")
def user_history(username):
    history = load_history()
    return jsonify([h for h in history if h["username"] == username])

# ---------- SAMPLE ----------
@app.route("/download-sample")
def download_sample():
    return jsonify({
        "message": "Use Google Drive link for dataset"
    })
