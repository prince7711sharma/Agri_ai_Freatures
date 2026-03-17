import joblib
import numpy as np
import os

model = None
scaler = None
encoder = None


# LOAD MODELS SAFELY
def load_models():
    global model, scaler, encoder

    if model is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model = joblib.load(os.path.join(BASE_DIR, "../models/irrigation_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "../models/irrigation_scaler.pkl"))
        encoder = joblib.load(os.path.join(BASE_DIR, "../models/status_encoder.pkl"))


# SAFE ENCODER (NO CRASH)
def safe_transform(encoder, value):
    value = value.lower()

    if value not in encoder.classes_:
        # fallback to first class
        return encoder.transform([encoder.classes_[0]])[0]

    return encoder.transform([value])[0]


# SMART RESPONSE
def irrigation_advice(label):
    mapping = {
        "Very Dry": {
            "action": "🚨 Immediate irrigation required",
            "water_level": "High"
        },
        "Dry": {
            "action": "💧 Irrigation needed soon",
            "water_level": "Medium"
        },
        "Wet": {
            "action": "✅ No irrigation needed",
            "water_level": "Low"
        },
        "Very Wet": {
            "action": "⚠ Avoid irrigation",
            "water_level": "Very Low"
        }
    }

    return mapping.get(label, {
        "action": "Unknown condition",
        "water_level": "Unknown"
    })


# MAIN FUNCTION
def predict_irrigation(data):
    load_models()

    status_encoded = safe_transform(encoder, data.status)

    features = np.array([[
        data.temperature,
        data.pressure,
        data.altitude,
        data.soil_moisture,
        status_encoded,
        data.hour,
        data.day,
        data.month
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    advice = irrigation_advice(prediction)

    return {
        "soil_condition": prediction,
        "recommendation": advice["action"],
        "water_level": advice["water_level"]
    }