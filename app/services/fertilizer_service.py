import joblib
import numpy as np

model = None
soil_encoder = None
crop_encoder = None
fert_encoder = None
scaler = None


def load_models():
    global model, soil_encoder, crop_encoder, fert_encoder, scaler

    if model is None:
        model = joblib.load("app/models/fertilizer_model.pkl")
        soil_encoder = joblib.load("app/models/soil_encoder.pkl")
        crop_encoder = joblib.load("app/models/crop_encoder.pkl")
        fert_encoder = joblib.load("app/models/fert_encoder.pkl")
        scaler = joblib.load("app/models/fertilizer_scaler.pkl")


def safe_transform(encoder, value):
    value = value.strip().lower()
    mapping = {cls.lower(): cls for cls in encoder.classes_}

    if value not in mapping:
        raise ValueError(f"{value} not found in training data")

    return encoder.transform([mapping[value]])[0]


def predict_fertilizer(data):
    load_models()

    soil = safe_transform(soil_encoder, data.soil_type)
    crop = safe_transform(crop_encoder, data.crop_type)

    features = np.array([[
        data.temperature,
        data.humidity,
        data.moisture,
        soil,
        crop,
        data.nitrogen,
        data.potassium,
        data.phosphorous
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    fertilizer_name = fert_encoder.inverse_transform([prediction])[0]

    return {
        "recommended_fertilizer": fertilizer_name
    }
