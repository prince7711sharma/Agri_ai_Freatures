# import joblib
# import numpy as np
#
# model = None
# state_encoder = None
# season_encoder = None
# crop_encoder = None
#
#
# def load_models():
#     global model, state_encoder, season_encoder, crop_encoder
#
#     if model is None:
#         model = joblib.load("app/models/production_model.pkl")
#         state_encoder = joblib.load("app/models/state_encoder.pkl")
#         season_encoder = joblib.load("app/models/season_encoder.pkl")
#         crop_encoder = joblib.load("app/models/crop_encoder.pkl")
#
#
# def safe_transform(encoder, value):
#     value = value.strip().lower()
#     mapping = {cls.lower(): cls for cls in encoder.classes_}
#
#     if value not in mapping:
#         raise ValueError(f"{value} not found in training data")
#
#     return encoder.transform([mapping[value]])[0]
#
#
# def predict_yield(data):
#     load_models()
#
#     state = safe_transform(state_encoder, data.state)
#     season = safe_transform(season_encoder, data.season)
#     crop = safe_transform(crop_encoder, data.crop)
#
#     features = np.array([[
#         state,
#         season,
#         crop,
#         data.year,
#         data.area
#     ]])
#
#     predicted_production = model.predict(features)[0]
#     predicted_yield = predicted_production / data.area
#
#     return {
#         "predicted_production": float(predicted_production),
#         "predicted_yield": float(predicted_yield)
#     }

import joblib
import numpy as np

model = None
state_encoder = None
season_encoder = None
crop_encoder = None


def load_models():
    global model, state_encoder, season_encoder, crop_encoder

    if model is None:
        model = joblib.load("app/models/new_production_model.pkl")
        state_encoder = joblib.load("app/models/state_encoder.pkl")
        season_encoder = joblib.load("app/models/season_encoder.pkl")
        crop_encoder = joblib.load("app/models/crop_encoder.pkl")


def normalize(text):
    return text.strip().lower()


def safe_transform(encoder, value):
    value = normalize(value)

    mapping = {normalize(cls): cls for cls in encoder.classes_}

    if value not in mapping:
        raise ValueError(
            f"{value} not found. Available values: {list(mapping.keys())[:5]}"
        )

    return encoder.transform([mapping[value]])[0]


def predict_yield(data):
    load_models()

    state = safe_transform(state_encoder, data.state)
    season = safe_transform(season_encoder, data.season)
    crop = safe_transform(crop_encoder, data.crop)

    features = np.array([[
        state,
        season,
        crop,
        data.year,
        data.area
    ]])

    predicted_production = model.predict(features)[0]
    predicted_yield = predicted_production / data.area

    return {
        "predicted_production": float(predicted_production),
        "predicted_yield": float(predicted_yield)
    }
