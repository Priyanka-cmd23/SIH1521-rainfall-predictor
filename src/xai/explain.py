"""Phase 4 (local): explain one individual prediction with SHAP."""
import os
import sys

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

sys.path.insert(0, os.path.join("src"))
from features.feature_config import MODEL_FEATURES  # noqa: E402

MODEL_PATH = os.path.join("models", "xgb_ens_0.json")
_STATE = {}


def get_model():
    if "model" not in _STATE:
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        _STATE["model"] = model
        _STATE["explainer"] = shap.TreeExplainer(model)
    return _STATE["model"], _STATE["explainer"]


def explain_local(row):
    """Return per-feature SHAP attributions for one prediction row."""
    _, explainer = get_model()
    X = pd.DataFrame([row])[MODEL_FEATURES].astype(float)
    values = explainer.shap_values(X, check_additivity=False)[0]
    items = []
    for feature, value, sh in zip(MODEL_FEATURES, X.iloc[0].values, values):
        items.append({
            "feature": feature,
            "value": round(float(value), 2),
            "shap": round(float(sh), 4),
            "direction": "increased" if sh >= 0 else "decreased",
            "impact": "pushed towards heavy rain" if sh >= 0 else "pushed away from heavy rain",
        })
    items.sort(key=lambda i: abs(i["shap"]), reverse=True)
    return items
