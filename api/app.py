"""Phase 6: FastAPI backend for the heavy-rainfall predictor.

Endpoints:
  POST /predict  - weather input -> prediction + probability + reliability + XAI
  GET  /metrics  - model performance + global SHAP importance
  GET  /features - feature definitions
  GET  /         - the web dashboard (static frontend)
"""
import datetime
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import requests
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features.feature_config import FEATURES, MODEL_FEATURES  # noqa: E402
from reliability.reliability import assess  # noqa: E402
from xai.explain import explain_local  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / "models"

app = FastAPI(title="Heavy Rainfall Predictor", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _load_models():
    members = []
    for i in range(3):
        m = xgb.XGBClassifier()
        m.load_model(str(MODELS_DIR / f"xgb_ens_{i}.json"))
        members.append(m)
    return members


_model_members = _load_models()

with open(MODELS_DIR / "metadata.json") as fh:
    _metadata = json.load(fh)
with open(MODELS_DIR / "metrics.json") as fh:
    _metrics = json.load(fh)
with open(MODELS_DIR / "global_shap.json") as fh:
    _global_shap = json.load(fh)

FEATURE_INFO = {f["name"]: f for f in FEATURES}

with open(BASE / "data" / "regions.json") as fh:
    _regions = json.load(fh)

WMO_CODES = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
             45: "Fog", 48: "Fog", 51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
             61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain",
             67: "Freezing rain", 71: "Slight snow", 73: "Snow", 75: "Heavy snow",
             80: "Rain showers", 81: "Rain showers", 82: "Violent showers",
             95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Thunderstorm + hail"}


def dew_point(temp_c, rh):
    """Magnus formula: dew point from air temperature and relative humidity."""
    a, b = 17.27, 237.7
    gamma = (a * temp_c) / (b + temp_c) + math.log(rh / 100.0)
    return (b * gamma) / (a - gamma)


def _predict_proba(row):
    X = np.array([[row[f] for f in MODEL_FEATURES]])
    probs = np.array([m.predict_proba(X)[:, 1][0] for m in _model_members])
    return float(np.mean(probs)), float(np.std(probs))


@app.get("/")
def root():
    return FileResponse(BASE / "frontend" / "index.html")


@app.get("/features")
def features():
    return {"model_features": MODEL_FEATURES, "features": FEATURES}


@app.get("/regions")
def regions():
    return _regions


@app.get("/forecast")
def forecast(city: str):
    """Live prediction for a city using Open-Meteo's FREE forecast API (no key).

    geocoding -> current weather -> feature mapping -> model -> XAI -> reliability.
    """
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=15,
        )
        geo.raise_for_status()
        results = geo.json().get("results") or []
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Geocoding service unavailable")
    if not results:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    loc = results[0]

    try:
        met = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,relative_humidity_2m,cloud_cover,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,weather_code",
                "daily": "precipitation_sum",
                "past_days": 1,
                "timezone": "Asia/Kolkata",
            },
            timeout=20,
        )
        met.raise_for_status()
        d = met.json()
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Forecast service unavailable")

    c = d["current"]
    t = float(c["temperature_2m"])
    row = {
        "temp_mean": t,
        "temp_max": t,
        "temp_min": t,
        "relative_humidity": float(c["relative_humidity_2m"]),
        "dew_point": round(dew_point(t, float(c["relative_humidity_2m"])), 2),
        "surface_pressure": float(c["surface_pressure"]),
        "cloud_cover": float(c["cloud_cover"]),
        "wind_speed_mean": float(c["wind_speed_10m"]),
        "wind_speed_max": float(c["wind_speed_10m"]),
        "wind_gust": float(c["wind_gusts_10m"]),
        "wind_direction": float(c["wind_direction_10m"]),
        "rain_lag1": float(d["daily"]["precipitation_sum"][0]),
        "day_of_year": int(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30))).timetuple().tm_yday),
    }
    prob, std = _predict_proba(row)
    reliability = assess(row, prob, ensemble_std=std)
    explanation = [
        {
            "feature": e["feature"],
            "label": FEATURE_INFO[e["feature"]]["why"][:60] + "...",
            "value": e["value"],
            "shap": e["shap"],
            "direction": e["direction"],
            "impact": e["impact"],
        }
        for e in explain_local(row)[:8]
    ]
    return {
        "city": loc["name"],
        "location": f"{loc['name']}, {loc.get('admin1', '')} ({loc.get('country_code', '')})",
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "current_weather_code": c.get("weather_code"),
        "current_conditions": WMO_CODES.get(c.get("weather_code"), "n/a"),
        "prediction": reliability["label"],
        "probability": round(prob, 4),
        "risk_level": "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW"),
        "reliability": reliability,
        "explanation": explanation,
        "model_agreement_std": round(std, 4),
        "data_source": "Open-Meteo live forecast (free, no API key)",
        "disclaimer": "Prototype demo: model was trained on Konkan-coast data. "
                      "Live forecast is a demonstration, not an operational product.",
    }


@app.get("/metrics")
def metrics():
    return {"metrics": _metrics, "global_shap": _global_shap}


@app.post("/predict")
def predict(payload: dict):
    missing = [f for f in MODEL_FEATURES if f not in payload or payload[f] is None]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    bad = [f for f in MODEL_FEATURES if not isinstance(payload[f], (int, float))]
    if bad:
        raise HTTPException(status_code=422, detail=f"Non-numeric values: {bad}")

    row = {f: float(payload[f]) for f in MODEL_FEATURES}
    prob, std = _predict_proba(row)

    explanation = explain_local(row)
    reliability = assess(row, prob, ensemble_std=std)
    explanation = explanation[:8]

    important = [
        {
            "feature": e["feature"],
            "label": FEATURE_INFO[e["feature"]]["why"][:60] + "...",
            "value": e["value"],
            "shap": e["shap"],
            "direction": e["direction"],
            "impact": e["impact"],
        }
        for e in explanation
    ]

    return {
        "prediction": reliability["label"],
        "probability": round(prob, 4),
        "risk_level": "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW"),
        "reliability": reliability,
        "explanation": important,
        "model_agreement_std": round(std, 4),
        "heavy_rain_threshold_mm": _metrics["heavy_rain_threshold_mm"],
        "disclaimer": "Prototype built on public IMD/ERA5 data (2001-2020, Konkan "
                      "coast). Not an operational ISRO forecasting system.",
    }


app.mount("/frontend", StaticFiles(directory=str(BASE / "frontend")), name="frontend")