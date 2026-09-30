from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
HOME_MODEL_PATH = os.path.join(BASE_DIR, "models", "home_model.joblib")
CLINIC_MODEL_PATH = os.path.join(BASE_DIR, "models", "clinic_model.joblib")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.json")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics_data.json")

home_model = joblib.load(HOME_MODEL_PATH) if os.path.exists(HOME_MODEL_PATH) else None
clinic_model = joblib.load(CLINIC_MODEL_PATH) if os.path.exists(CLINIC_MODEL_PATH) else None

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

with open(METRICS_PATH, "r") as f:
    metrics_data = json.load(f)

def get_risk_band(prob, threshold):
    low_cut = threshold / 2.0
    if prob >= threshold:
        return "HIGH"
    elif prob >= low_cut:
        return "MODERATE"
    return "LOWER"

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "home_model_loaded": home_model is not None, "clinic_model_loaded": clinic_model is not None})

@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify(config)

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics_data)

@app.route("/api/predict-home", methods=["POST"])
def predict_home():
    if not home_model:
        return jsonify({"error": "Home model not loaded"}), 500

    data = request.json or {}
    input_df = pd.DataFrame([{
        "age": float(data.get("age", 55)),
        "sex": str(data.get("sex", "Male")),
        "cp": str(data.get("cp", "asymptomatic")),
        "trestbps": float(data.get("trestbps", 130)) if data.get("trestbps") else np.nan,
        "exang": str(data.get("exang", "False")),
        "chol": float(data.get("chol")) if data.get("chol") is not None and str(data.get("chol")).strip() != "" else np.nan,
        "fbs": str(data.get("fbs")) if data.get("fbs") in ["True", "False"] else np.nan
    }])

    prob = float(home_model.predict_proba(input_df)[0, 1])
    thr = config.get("home_threshold", 0.43)
    band = get_risk_band(prob, thr)

    return jsonify({
        "mode": "home",
        "probability": round(prob, 4),
        "threshold": thr,
        "risk_level": band,
        "disease_predicted": bool(prob >= thr)
    })

@app.route("/api/predict-clinic", methods=["POST"])
def predict_clinic():
    if not clinic_model:
        return jsonify({"error": "Clinic model not loaded"}), 500

    data = request.json or {}
    input_df = pd.DataFrame([{
        "age": float(data.get("age", 58)),
        "sex": str(data.get("sex", "Male")),
        "cp": str(data.get("cp", "asymptomatic")),
        "trestbps": float(data.get("trestbps", 140)) if data.get("trestbps") else np.nan,
        "exang": str(data.get("exang", "True")),
        "chol": float(data.get("chol", 240)) if data.get("chol") else np.nan,
        "fbs": str(data.get("fbs", "False")),
        "restecg": str(data.get("restecg", "normal")),
        "thalch": float(data.get("thalch", 135)) if data.get("thalch") else np.nan,
        "oldpeak": float(data.get("oldpeak", 1.5)) if data.get("oldpeak") is not None else np.nan,
        "slope": str(data.get("slope", "flat"))
    }])

    prob = float(clinic_model.predict_proba(input_df)[0, 1])
    thr = config.get("clinic_threshold", 0.44)
    band = get_risk_band(prob, thr)

    return jsonify({
        "mode": "clinic",
        "probability": round(prob, 4),
        "threshold": thr,
        "risk_level": band,
        "disease_predicted": bool(prob >= thr)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
