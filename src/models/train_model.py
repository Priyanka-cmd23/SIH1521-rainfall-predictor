"""Phase 3: Train the XGBoost heavy-rainfall classifier (small ensemble).

Train -> val -> test via time-aware split. Prediction is the mean probability
of 3 seeds (ensemble disagreement feeds the reliability module).

Outputs in models/:
  xgb_ens_{0,1,2}.json  - trained XGBoost models (xgboost native format)
  metrics.json          - precision/recall/F1/ROC-AUC/CM + POD/FAR/CSI
  metadata.json         - feature order, training mean/std (for OOD checks),
                          class balance, threshold
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

import sys

sys.path.insert(0, os.path.join("src"))
from features.feature_config import MODEL_FEATURES, TARGET, THRESHOLD_MM  # noqa: E402

SPLIT_DIR = os.path.join("data", "processed", "split")
MODEL_DIR = "models"
SEEDS = [42, 7, 2020]
PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
              use_label_encoder=False, verbosity=0, n_jobs=-1)


def categorical_metrics(y, p):
    """Standard classification metrics + meteorology POD/FAR/CSI."""
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = f1_score(y, p)
    pod = rec
    far = fp / (tp + fp) if tp + fp else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    auc = roc_auc_score(y, p)
    acc = accuracy_score(y, p)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc,
                pod=pod, far=far, csi=csi,
                confusion_matrix={"tn": int(tn), "fp": int(fp),
                                  "fn": int(fn), "tp": int(tp)},
                n=len(y), n_heavy=int(y.sum()))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))

    X_tr, y_tr = train[MODEL_FEATURES], train[TARGET].values
    X_val, y_val = val[MODEL_FEATURES], val[TARGET].values
    X_te, y_te = test[MODEL_FEATURES], test[TARGET].values

    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    print(f"[train] {len(X_tr):,} rows | positives {pos} | scale_pos_weight {scale_pos_weight:.2f}")

    models = []
    for seed in SEEDS:
        model = xgb.XGBClassifier(**PARAMS, random_state=seed, scale_pos_weight=scale_pos_weight)
        model.fit(X_tr, y_tr, verbose=False)
        path = os.path.join(MODEL_DIR, f"xgb_ens_{SEEDS.index(seed)}.json")
        model.save_model(path)
        models.append(model)
        print(f"[train] seed {seed} done -> {path}")

    def ens_prob(X):
        return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

    probs_val = ens_prob(X_val)
    probs_test = ens_prob(X_test := X_te)

    pred_thr = 0.5
    metrics = {
        "threshold_prediction": pred_thr,
        "heavy_rain_threshold_mm": THRESHOLD_MM,
        "val": categorical_metrics(y_val, (probs_val >= pred_thr).astype(int)),
        "test": categorical_metrics(y_te, (probs_test >= pred_thr).astype(int)),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    metadata = {
        "model_features": MODEL_FEATURES,
        "seeds": SEEDS,
        "scale_pos_weight": float(scale_pos_weight),
        "n_train": int(len(X_tr)),
        "feature_mean": {c: float(X_tr[c].mean()) for c in MODEL_FEATURES},
        "feature_std": {c: float(X_tr[c].std()) for c in MODEL_FEATURES},
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print("\n== VAL ==")
    for k, v in metrics["val"].items():
        if not isinstance(v, dict):
            print(f"  {k:<16} {v:.4f}")
    print("\n== TEST ==")
    for k, v in metrics["test"].items():
        if not isinstance(v, dict):
            print(f"  {k:<16} {v:.4f}")
    print("\n[saved] models + metrics.json + metadata.json")


if __name__ == "__main__":
    main()