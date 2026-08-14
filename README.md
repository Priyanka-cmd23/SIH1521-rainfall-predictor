# 🌧️ XAI-based Heavy Rainfall Predictor

A beginner-friendly prototype that predicts the probability of heavy rainfall
from atmospheric data, explains **why** with SHAP, scores prediction reliability,
and shows everything on a web dashboard.

> **Prototype built using publicly available atmospheric/rainfall data, with an
> INSAT-compatible pipeline designed for future integration.**

---

## Problem statement
Many lives and livelihoods depend on warnings of very heavy rainfall. ISRO
wants a system that predicts heavy rain and — just as importantly — **explains
its reasoning** and flags when the prediction may be unreliable.

## Project objective
Complete pipeline: **DATA → PREPROCESSING → FEATURES → ML MODEL → PREDICTION →
XAI → RELIABILITY → API → WEB DASHBOARD**

## System architecture
```
IMD 0.25° daily rainfall (target)   ERA5 atmosphere via Open-Meteo (features)
          │                                      │
          └──────────────┬───────────────────────┘
                 data/processed/rainfall_dataset.csv
                 time-aware train/val/test split (2001–2020, Konkan coast)
                 XGBoost ensemble (3 seeds) -> SHAP -> reliability
                          │
                   FastAPI  (/predict /metrics /features)
                          │
            Static dashboard (HTML/CSS/JS, no build step)
```

## Dataset
| Role | Source | Detail | Cost |
|---|---|---|---|
| **Target** | IMD 0.25° daily gridded rainfall (Pai et al. 2014) | Gauge-based, India, 2001–2020 | Free |
| **Features** | ERA5 reanalysis via Open-Meteo archive API | 0.25°, daily aggregations | Free, no key |

Region: **Konkan coast** latitudes 15.5–20.0 °N (Mumbai→Goa), so heavy-rain events
are common enough to learn from. Final dataset: **343,294 rows · 41 grid cells ·
2001-01-01 → 2020-12-31 · heavy-rain days 2.24%**.

Origin scripts: `src/data/download_imd_rainfall.py`, `download_era5_features.py`,
`build_dataset.py` (all resumable — rerun them to add more cells/years).

## Target definition
`1` if IMD daily rainfall **> 64.5 mm** else `0`. 64.5 mm is IMD's own "Very
Heavy Rain" lower category boundary. Kept as a config constant
(`THRESHOLD_MM`) so multi-class categories can be added later.

## Features (registry: `src/features/feature_config.py`)
13 model features, each with units + physical reason: `temp_mean/max/min`,
`relative_humidity`, `dew_point`, `surface_pressure`, `cloud_cover`,
`wind_speed_mean/max`, `wind_gust`, `wind_direction`, `rain_lag1` (previous day's
rain — the one truly *predictive* feature), `day_of_year` (monsoon seasonality).
Lat/lon are kept for spatial reference but excluded from the model so SHAP
explanations stay physical. Same-day features make the MVP a **diagnostic /
nowcast** rather than a true 24 h forecast — this is documented in Limitations.

## Model
**XGBoost** binary classifier (CPU-friendly, SHAP-compatible). Class imbalance
handled with `scale_pos_weight` (~48:1). An **ensemble of 3 seeds** gives mean
probability + disagreement signal. Time-aware split:
train 2001–2016 · val 2017–2018 · test 2019–2020 (no future leakage).

## Evaluation (test set)
| Metric | Value | Meaning |
|---|---|---|
| ROC-AUC | 0.923 | Model ranks heavy vs normal days well |
| Recall / POD | 0.915 | Catches 91.5% of real heavy-rain days |
| Precision | 0.359 | 36% of "heavy" warnings are true |
| F1 / CSI | 0.516 / 0.347 | Combined skill score |
| FAR | 0.641 | False alarms are high (rare events) |

Heavy rain is rare (~2%), so high recall comes with false alarms — exactly why
the reliability module exists. Metrics computed at a fixed 0.5 threshold; the
threshold can be tuned for stricter/looser alerts.

## XAI (SHAP)
- **Global:** `models/global_shap.json` — mean |SHAP| per feature. Top drivers:
  `rain_lag1`, `relative_humidity`, `cloud_cover`, `dew_point` (moisture + persistence).
- **Local:** POST `/predict` returns each feature's SHAP value and impact
  ("*relative_humidity pushed towards heavy rain*"). Values are real model
  attributions — never invented text.

## Reliability methodology
Three signals combined into **HIGH / MEDIUM / LOW**:
1. **Probability** → confidence = distance from 0.5.
2. **Out-of-distribution (OOD)** check → standardised distance of each input
   feature from training mean/std.
3. **Ensemble disagreement** → std of the 3 model probabilities.

OOD or strong disagreement downgrades reliability and adds a human-readable
reason. This is a *heuristic confidence indicator*, **not** a calibrated
probability of correctness.

## API (FastAPI)
```bash
.venv\Scripts\python.exe -m uvicorn api.app:app --port 8000
```
| Endpoint | Purpose |
|---|---|
| `POST /predict` | weather JSON → prediction, probability, risk, reliability, SHAP explanation |
| `GET /metrics` | model performance + global SHAP importance |
| `GET /features` | feature definitions |
| `GET /` | web dashboard |

### Live forecast — free Open-Meteo API (no key)
`GET /forecast?city=Bhubaneswar` geocodes the city (free geocoding API), pulls
**real-time** current weather (free forecast API), maps it to the model's 13
features (dew point computed by Magnus formula), and returns prediction +
reliability + SHAP. Open the dashboard → type any city → "Go". This makes the
demo live with real weather, still fully free.

### Multi-region support
`data/regions.json` defines state/coast boxes: **Maharashtra (Konkan)**,
**Odisha coast**, **Kerala coast**, plus 5 demo cities (Mumbai, Ratnagiri,
Bhubaneswar, Puri, Kochi). To build training data for another region:
```powershell
$env:REGION = "odisha_coast"
python src\data\download_era5_features.py   # resumable; add cells anytime
python src\data\build_dataset.py            # -> rainfall_dataset_odisha_coast.csv
```
The currently trained/served model is the Konkan-coast one; the same pipeline
retrains per region. Live forecast reuses it at any city as a demonstration.

## Frontend
Single self-contained page (`frontend/index.html`, no build step, no CDN deps).
Shows: prediction card, SHAP bar chart, reliability panel, model performance +
confusion matrix, 3 demo input buttons (heavy day / normal day / unusual input),
and a clearly-labelled **INSAT-3D/3DR Integration Ready** section.

## Installation & run locally
```powershell
cd C:\Users\radhi\rainfall-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt   # full dev stack (data + model + API)
# deployment-only (Vercel uses just this): python -m pip install -r requirements.txt

# 1. build the dataset (skips files already downloaded)
python src\data\download_imd_rainfall.py
python src\data\download_era5_features.py
python src\data\build_dataset.py

# 2. split + train + XAI
python src\features\prepare_split.py
python src\models\train_model.py
python src\xai\build_shap.py

# 3. run
python -m uvicorn api.app:app --port 8000
# open http://127.0.0.1:8000
```

Example API call:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" ^
  -d @data/sample/sample_inputs.json
```

## Testing
```bash
.\.venv\Scripts\python.exe -m pytest tests -q
```
Covers heavy/normal/unusual cases, missing and non-numeric inputs, metrics,
features and dashboard endpoints (8 tests).

## Limitations
- MVP is trained on **public IMD + ERA5 data**, not INSAT imagery.
- Same-day ERA5 features ⇒ diagnostic/nowcast, not an operational 24 h forecast.
- ~3 h offset exists between IMD's India-standard rain day and the ERA5 day;
  matched on calendar date.
- Precision is limited by the rarity of heavy rain; reliability flags (but
  cannot fix) uncertain cases.
- Trained on the Konkan coast; performance elsewhere would need retraining.

## Future INSAT-3D/3DR integration
Add satellite-derived columns (brightness temperature, cloud-top temperature,
water-vapour channels) as new entries in `feature_config.FEATURES`, retrain, and
the API/dashboard/XAI/reliability adapt automatically.

## What this prototype does NOT claim
- It is **not** a production operational ISRO forecasting system.
- It does **not** use real-time satellite data or display fake satellite data.
- Reliability is **not** a scientifically calibrated probability of correctness.
- Results are for the chosen region/period only and must be re-validated before
  any operational use.

## License / data credits
Code: open source. Data: IMD (Pai et al. 2014, cite in any output), ERA5
(Copernicus Climate Change Service) served by Open-Meteo (free, non-commercial).