# src/predict.py
# Prediction functions and utilities

import os
import math
import joblib
import numpy as np
import pandas as pd

# Paths to model artifacts
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "taxi_duration_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "selected_features.pkl")

# Lazy-loaded artifacts
_model = None
_scaler = None
_selected_features = None
_model_err = None


def load_artifacts():
    """Lazy-load model artifacts. Keep going even if missing."""
    global _model, _scaler, _selected_features, _model_err
    if _model is not None or _model_err is not None:
        return

    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH):
            _model = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
            _selected_features = joblib.load(FEATURES_PATH)
        else:
            _model_err = "Model files not found in /models. Prediction will be disabled."
    except Exception as e:
        _model_err = f"Could not load model artifacts: {e}"


def predict_duration(df_prep):
    """Predict trip duration in seconds"""
    global _model, _scaler, _selected_features, _model_err

    if _model_err:
        st.warning(_model_err)
        return None

    if _model is None or _scaler is None or _selected_features is None:
        return None

    # Ensure we have all selected features
    for f in _selected_features:
        if f not in df_prep.columns:
            df_prep[f] = 0

    X = df_prep[_selected_features].fillna(0)
    Xs = _scaler.transform(X)
    pred_log = _model.predict(Xs)[0]
    return float(np.expm1(pred_log))


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance between two points"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
        dlon / 2.0) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def estimate_co2_saving(orig_km, optim_km):
    """
    Estimate CO₂ savings based on distance reduction
    Formula: CO₂ saved (kg) = (original_km - optimized_km) * emission_factor
    Emission factor: 0.15 kg CO₂ per km (typical for NYC taxis)
    """
    emission_factor = 0.15  # kg CO₂ per km
    if isinstance(orig_km, (float, int)) and isinstance(optim_km, (float, int)):
        return (orig_km - optim_km) * emission_factor
    else:
        # Handle array inputs
        return (np.array(orig_km) - np.array(optim_km)) * emission_factor