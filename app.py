"""
app.py — Crop Advisor AI Backend
=================================
Predicts the best crop to grow based on soil & weather data.
Uses Random Forest model + Gemini LLM for farming advice.
"""

import os, json, csv, logging
from datetime import datetime

import joblib
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ── Config ─────────────────────────────────────────────────────────────────────
app    = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "AIzaSyDIT3-ueLvy9ATbI4UIPQ9T7bWUxFMm-hg")
MODEL_PATH       = os.path.join("model", "crop_model.pkl")
SCALER_PATH      = os.path.join("model", "scaler.pkl")
METADATA_PATH    = os.path.join("model", "model_metadata.json")
LOG_FILE         = "predictions_log.csv"
FEATURE_COLS     = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ── Crop Knowledge Base ────────────────────────────────────────────────────────
CROP_INFO = {
    "rice":        {"season":"Kharif", "water":"High",   "sdg":"SDG 2", "emoji":"🌾"},
    "maize":       {"season":"Kharif", "water":"Medium", "sdg":"SDG 2", "emoji":"🌽"},
    "chickpea":    {"season":"Rabi",   "water":"Low",    "sdg":"SDG 2", "emoji":"🫘"},
    "kidneybeans": {"season":"Kharif", "water":"Medium", "sdg":"SDG 2", "emoji":"🫘"},
    "pigeonpeas":  {"season":"Kharif", "water":"Low",    "sdg":"SDG 2", "emoji":"🌿"},
    "mothbeans":   {"season":"Kharif", "water":"Low",    "sdg":"SDG 15","emoji":"🌱"},
    "mungbean":    {"season":"Kharif", "water":"Low",    "sdg":"SDG 2", "emoji":"🫘"},
    "blackgram":   {"season":"Kharif", "water":"Low",    "sdg":"SDG 2", "emoji":"🫘"},
    "lentil":      {"season":"Rabi",   "water":"Low",    "sdg":"SDG 2", "emoji":"🫘"},
    "pomegranate": {"season":"Annual", "water":"Low",    "sdg":"SDG 3", "emoji":"🍎"},
    "banana":      {"season":"Annual", "water":"High",   "sdg":"SDG 2", "emoji":"🍌"},
    "mango":       {"season":"Annual", "water":"Medium", "sdg":"SDG 3", "emoji":"🥭"},
    "grapes":      {"season":"Annual", "water":"Medium", "sdg":"SDG 3", "emoji":"🍇"},
    "watermelon":  {"season":"Summer", "water":"High",   "sdg":"SDG 3", "emoji":"🍉"},
    "muskmelon":   {"season":"Summer", "water":"Medium", "sdg":"SDG 3", "emoji":"🍈"},
    "apple":       {"season":"Annual", "water":"Medium", "sdg":"SDG 3", "emoji":"🍎"},
    "orange":      {"season":"Annual", "water":"Medium", "sdg":"SDG 3", "emoji":"🍊"},
    "papaya":      {"season":"Annual", "water":"Medium", "sdg":"SDG 3", "emoji":"🍈"},
    "coconut":     {"season":"Annual", "water":"Medium", "sdg":"SDG 2", "emoji":"🥥"},
    "cotton":      {"season":"Kharif", "water":"Medium", "sdg":"SDG 9", "emoji":"☁️"},
    "jute":        {"season":"Kharif", "water":"High",   "sdg":"SDG 9", "emoji":"🌿"},
    "coffee":      {"season":"Annual", "water":"Medium", "sdg":"SDG 2", "emoji":"☕"},
}

# ── Load Model ─────────────────────────────────────────────────────────────────
def load_model():
    try:
        model    = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        logger.info(f"✅ Model loaded — {meta['model_name']} ({meta['accuracy']}% accuracy)")
        return model, scaler, meta
    except FileNotFoundError:
        logger.warning("⚠️  Model files not found in model/ folder.")
        logger.warning("    Train first on Colab, then download and place in model/")
        return None, None, {"model_name":"Not loaded","accuracy":0,"class_names":list(CROP_INFO.keys())}

model, scaler, metadata = load_model()

# ── Gemini Setup ───────────────────────────────────────────────────────────────
def setup_gemini():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return None

gemini = setup_gemini()

def log_prediction(inputs, crop, confidence):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","N","P","K","temperature","humidity","ph","rainfall","crop","confidence"])
        row = [datetime.now().isoformat()] + [inputs.get(c,"-") for c in FEATURE_COLS] + [crop, f"{confidence:.1f}"]
        w.writerow(row)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "model_name":   metadata.get("model_name","—"),
        "accuracy":     metadata.get("accuracy", 0),
        "num_classes":  len(metadata.get("class_names",[])),
        "timestamp":    datetime.now().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict best crop from soil & weather inputs.
    Expects JSON: {"N":90,"P":42,"K":43,"temperature":20.8,"humidity":82.0,"ph":6.5,"rainfall":202.9}
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Validate inputs
    missing = [c for c in FEATURE_COLS if c not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        features = np.array([[float(data[c]) for c in FEATURE_COLS]])

        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded. Train on Colab first."}), 503

        features_scaled = scaler.transform(features)

        # Predict
        prediction     = model.predict(features_scaled)[0]
        probabilities  = model.predict_proba(features_scaled)[0]
        confidence     = float(np.max(probabilities) * 100)

        class_names    = metadata.get("class_names", list(CROP_INFO.keys()))
        crop_name      = class_names[prediction] if isinstance(prediction, (int, np.integer)) else str(prediction)

        # Top 3 crops
        top3_idx  = np.argsort(probabilities)[::-1][:3]
        top3      = [{"crop": class_names[i], "confidence": round(probabilities[i]*100, 1)}
                     for i in top3_idx]

        # Extra info
        info = CROP_INFO.get(crop_name.lower(), {"season":"—","water":"—","sdg":"SDG 2","emoji":"🌱"})

        log_prediction(data, crop_name, confidence)

        return jsonify({
            "success":    True,
            "crop":       crop_name,
            "confidence": round(confidence, 1),
            "top3":       top3,
            "info":       info,
            "inputs":     {c: data[c] for c in FEATURE_COLS},
            "timestamp":  datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/advise", methods=["POST"])
def advise():
    """Get Gemini LLM farming advice for recommended crop."""
    data = request.get_json()
    if not data or "crop" not in data:
        return jsonify({"error": "Missing crop field"}), 400

    crop   = data.get("crop","")
    inputs = data.get("inputs", {})
    info   = CROP_INFO.get(crop.lower(), {})

    prompt = f"""You are an expert agricultural scientist advising a farmer in India.

The AI system analyzed the farmer's soil and weather data:
- Nitrogen (N): {inputs.get('N','—')} kg/ha
- Phosphorus (P): {inputs.get('P','—')} kg/ha
- Potassium (K): {inputs.get('K','—')} kg/ha
- Temperature: {inputs.get('temperature','—')}°C
- Humidity: {inputs.get('humidity','—')}%
- Soil pH: {inputs.get('ph','—')}
- Rainfall: {inputs.get('rainfall','—')} mm

Recommended Crop: **{crop.upper()}** (Season: {info.get('season','—')}, Water Need: {info.get('water','—')})

Provide a practical advisory with these sections:
1. WHY THIS CROP (2 sentences explaining why soil/weather suits this crop)
2. PLANTING GUIDE (when to plant, row spacing, seed rate)
3. FERTILIZER SCHEDULE (NPK dosage based on given N={inputs.get('N','—')}, P={inputs.get('P','—')}, K={inputs.get('K','—')})
4. WATER MANAGEMENT (irrigation frequency for {inputs.get('rainfall','—')}mm rainfall)
5. EXPECTED YIELD & MARKET PRICE (approximate per hectare)
6. SDG IMPACT (how growing {crop} supports {info.get('sdg','SDG 2')} - Zero Hunger)

Keep response under 350 words. Use simple, practical language."""

    try:
        if gemini:
            response = gemini.generate_content(prompt)
            advice   = response.text
            source   = "Gemini 1.5 Flash"
        else:
            advice = fallback_advice(crop)
            source = "Built-in Database"

        return jsonify({
            "success":   True,
            "crop":      crop,
            "advice":    advice,
            "source":    source,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success":   True,
            "crop":      crop,
            "advice":    fallback_advice(crop),
            "source":    "Built-in Database (Gemini error)",
            "timestamp": datetime.now().isoformat()
        })


def fallback_advice(crop):
    return f"""**WHY THIS CROP:** Your soil nutrients and climate conditions are optimal for {crop.title()} cultivation. The temperature, humidity, and rainfall parameters closely match the ideal growing environment.

**PLANTING GUIDE:** Plant during the appropriate season. Maintain row spacing of 30-45cm. Use 15-20 kg/ha of quality seeds. Prepare land 2-3 weeks before sowing.

**FERTILIZER SCHEDULE:** Apply basal dose at sowing: 50% N + full P + full K. Top-dress remaining N in 2 splits (at 30 and 60 days). Organic manure (FYM) 5 tonnes/ha recommended.

**WATER MANAGEMENT:** Irrigate at critical growth stages. Avoid waterlogging. Drip irrigation saves 40% water and improves yield.

**EXPECTED YIELD:** 2.5-4 tonnes/ha depending on variety and management. Market price varies by season — check local mandi rates.

**SDG IMPACT:** Growing {crop.title()} helps achieve SDG 2 (Zero Hunger) by increasing food production and farmer income while maintaining soil health for future generations."""


@app.route("/history")
def history():
    records = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            reader = csv.DictReader(f)
            records = list(reader)[-15:]
    return jsonify({"history": records})


@app.route("/crops")
def crops():
    return jsonify({"crops": list(CROP_INFO.keys()), "total": len(CROP_INFO)})


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Crop Advisor AI starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
