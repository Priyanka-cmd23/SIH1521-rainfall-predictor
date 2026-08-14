"""Phase 4 (local): explain one individual prediction with SHAP.

Uses XGBoost's native TreeSHAP (pred_contribs) so the server needs NO shap
package and NO pandas - keeping the deployment small. The SHAP values are the
same ones the shap package would produce.
"""
import os
import sys

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.join("src"))
from features.feature_config import MODEL_FEATURES  # noqa: E402

MODEL_PATH = os.path.join("models", "xgb_ens_0.json")
_STATE = {}


def get_model():
    if "model" not in _STATE:
        booster = xgb.Booster()
        booster.load_model(MODEL_PATH)
        _STATE["model"] = booster
    return _STATE["model"]


def explain_local(row):
    """Return per-feature SHAP attributions via XGBoost native TreeSHAP."""
    model = get_model()
    X = np.array([[float(row[f]) for f in MODEL_FEATURES]])
    dmat = xgb.DMatrix(X, feature_names=MODEL_FEATURES)
    contribs = model.predict(dmat, pred_contribs=True)[0]

    items = []
    for feature, value, sh in zip(MODEL_FEATURES, X[0], contribs[:-1]):
        items.append({
            "feature": feature,
            "value": round(float(value), 2),
            "shap": round(float(sh), 4),
            "direction": "increased" if sh >= 0 else "decreased",
            "impact": "pushed towards heavy rain" if sh >= 0 else "pushed away from heavy rain",
        })
    items.sort(key=lambda i: abs(i["shap"]), reverse=True)
    return items