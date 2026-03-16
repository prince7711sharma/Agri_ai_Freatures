import joblib
import numpy as np

model = None


def load_model():
    global model
    if model is None:
        model = joblib.load("app/models/crop_model.pkl")


def predict_crop(data):
    load_model()

    features = np.array([[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]])

    prediction = model.predict(features)[0]

    return {
        "recommended_crop": prediction
    }