"""Phase 4: Global SHAP feature importance.

Loads a sample of training data, computes SHAP values with the first ensemble
member, and saves the global feature importance to models/global_shap.json.
"""
import json
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

import sys

sys.path.insert(0, os.path.join("src"))
from features.feature_config import MODEL_FEATURES, TARGET  # noqa: E402

SPLIT_DIR = os.path.join("data", "processed", "split")
MODEL_PATH = os.path.join("models", "xgb_ens_0.json")
OUT = os.path.join("models", "global_shap.json")
SAMPLE = 20000


def main():
    train = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    pos = train[train[TARGET] == 1]
    neg = train[train[TARGET] == 0].sample(n=min(len(pos) * 5, SAMPLE - len(pos)), random_state=1)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=1).reset_index(drop=True)
    X = sample[MODEL_FEATURES]
    print(f"[sample] {len(X)} rows ({int(sample[TARGET].sum())} heavy)")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)

    imp = np.abs(shap_values).mean(axis=0)
    rows = [{"feature": f, "importance": float(v)} for f, v in zip(MODEL_FEATURES, imp)]
    rows.sort(key=lambda r: r["importance"], reverse=True)

    with open(OUT, "w") as fh:
        json.dump({"base_value": float(explainer.expected_value), "importance": rows}, fh, indent=2)

    print("== global feature importance (mean |SHAP|) ==")
    for r in rows:
        print(f"  {r['feature']:<20} {r['importance']:.4f}")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()