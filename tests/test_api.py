"""Phase 9: automated API tests.

Run with:  .venv\\Scripts\\python.exe -m pytest tests -q
"""
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402
from src.features.feature_config import MODEL_FEATURES  # noqa: E402

client = TestClient(app)

with open(BASE / "data" / "sample" / "sample_inputs.json") as fh:
    CASES = json.load(fh)["cases"]


def _full(payload):
    row = dict(payload)
    for f in MODEL_FEATURES:
        if f not in row:
            row[f] = 50.0
    return row


def test_heavy_rain_case():
    r = client.post("/predict", json=_full(CASES["heavy_rain_day"]))
    assert r.status_code == 200
    d = r.json()
    assert d["probability"] > 0.7, d
    assert "rel" not in d  # sanity: keys exist
    assert "explanation" in d and len(d["explanation"]) > 0
    assert "reliability" in d
    assert d["reliability"]["prediction"] == 1


def test_normal_day_case():
    r = client.post("/predict", json=_full(CASES["normal_day"]))
    d = r.json()
    assert r.status_code == 200
    assert d["probability"] < 0.4, d


def test_unusual_input_flags_ood():
    r = client.post("/predict", json=_full(CASES["unusual_oceanic_input"]))
    d = r.json()
    assert r.status_code == 200
    assert d["reliability"]["ood_flag"] is True, d


def test_missing_feature_returns_422():
    bad = dict(_full(CASES["normal_day"]))
    bad.pop("cloud_cover")
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_non_numeric_returns_422():
    bad = _full(CASES["normal_day"])
    bad["cloud_cover"] = "abc"
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    d = r.json()
    assert "metrics" in d and "global_shap" in d
    assert "test" in d["metrics"]
    assert d["metrics"]["test"]["roc_auc"] > 0.8


def test_features_endpoint():
    r = client.get("/features")
    assert r.status_code == 200
    assert len(r.json()["model_features"]) == len(MODEL_FEATURES)


def test_regions_endpoint():
    r = client.get("/regions")
    assert r.status_code == 200
    regions = r.json()["regions"]
    assert "maharashtra_konkan" in regions and "odisha_coast" in regions
    assert "cities" in r.json() and len(r.json()["cities"]) >= 5


def test_dashboard_overview():
    r = client.get("/")
    assert r.status_code == 200
    assert "Heavy Rainfall Predictor" in r.text