import os
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from auth import register_user, authenticate_user
from detector import detect_anomaly

# -------------------- APP --------------------
app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": [
            "https://funny-genie-c4a8c1.netlify.app",
            "https://*.netlify.app"
        ]
    }
})
app.config["CORS_HEADERS"] = "Content-Type"

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# -------------------- HISTORY HELPERS --------------------
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# -------------------- ROUTES --------------------

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

    username = request.form.get("username", "unknown")
    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    result = detect_anomaly(image_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    history = load_history()
    BASE_URL = request.host_url.rstrip("/")
     
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
def get_uploaded_image(filename):
    return send_from_directory("uploads", filename)

@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# ---------- HISTORY ----------
@app.route("/history/<username>")
def user_history(username):
    history = load_history()
    return jsonify([h for h in history if h["username"] == username])

# ---------- SAMPLE DATA ----------
@app.route("/download-sample")
def download_sample():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "sample_data", "sample_data.zip")

    if not os.path.exists(file_path):
        return {"error": "Sample dataset not found"}, 404

    return send_file(file_path, as_attachment=True)

# -------------------- RUN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
