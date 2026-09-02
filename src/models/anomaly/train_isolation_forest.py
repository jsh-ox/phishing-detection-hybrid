"""
Anomaly Layer — Isolation Forest (G2 v2)
==============================================================================
Project: Hybrid ML for Phishing Detection

Trains an Isolation Forest on URLs (24 enriched features, label convention 0=phishing/
1=legitimate).

Outputs (into OUTPUT_DIR):
  isolation_forest.pkl        trained model
  if_scores_test.parquet      per-row test anomaly scores + label (for fusion)
  if_eval_report.json         metrics under all three threshold strategies
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from anomaly_eval import to_phishing_target, evaluate_all_strategies

# --- CONFIG ---
DATA_DIR   = Path("data") / "processed" / "G2_v2_nohttps"
TRAIN_PATH = DATA_DIR / "url_features_train.parquet"
VAL_PATH   = DATA_DIR / "url_features_val.parquet"
TEST_PATH  = DATA_DIR / "url_features_test.parquet"
OUTPUT_DIR = Path("results/anomaly/isolation_forest_nohttps")

LABEL_COL     = "label"
SEED          = 42
N_ESTIMATORS  = 200
MAX_SAMPLES   = 512
CONTAMINATION = "auto"
TARGET_FPR    = 0.10
# ======================================================


def load_features(path):
    """Return (feature_matrix, feature_names, labels_or_None)."""
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c != LABEL_COL]
    X = df[feats].values.astype(float)
    y = df[LABEL_COL].values if LABEL_COL in df.columns else None
    return X, feats, y


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, feats, y_train_raw = load_features(TRAIN_PATH)
    X_val, _, y_val_raw = load_features(VAL_PATH)
    X_test, _, y_test_raw = load_features(TEST_PATH)

    y_val  = to_phishing_target(y_val_raw)
    y_test = to_phishing_target(y_test_raw)

    ms = MAX_SAMPLES
    if isinstance(ms, str) and ms != "auto":
        ms = float(ms) if "." in ms else int(ms)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=ms,
        contamination=CONTAMINATION,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train)

    # anomaly scores: higher = more anomalous
    val_scores  = -model.score_samples(X_val)
    test_scores = -model.score_samples(X_test)

    results = evaluate_all_strategies(val_scores, y_val, test_scores, y_test, TARGET_FPR)

    joblib.dump(model, OUTPUT_DIR / "isolation_forest.pkl")
    pd.DataFrame({"anomaly_score": test_scores, "label": y_test_raw}).to_parquet(
        OUTPUT_DIR / "if_scores_test.parquet", index=False)

    report = {
        "model": "IsolationForest",
        "corpus": "G2_v2",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "label_convention": "0=phishing, 1=legitimate; detection target = phishing",
        "train_regime": "Contaminated URLS (mixed legitimate and phishing)",
        "n_features": len(feats),
        "hyperparameters": {"n_estimators": N_ESTIMATORS, "max_samples": ms,
                            "contamination": CONTAMINATION},
        "train_rows": len(X_train),
        "test_metrics_by_strategy": results,
    }
    with open(OUTPUT_DIR / "if_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # console summary
    print("=" * 60)
    print("ISOLATION FOREST — G2 v2")
    print("=" * 60)
    print(f"  features        : {len(feats)}")
    print(f"  train: {len(X_train):,}")
    print(f"  n_estimators    : {N_ESTIMATORS} | max_samples: {ms}")
    print("-" * 60)
    print(f"  {'strategy':<10}{'F1':>8}{'AUC-ROC':>10}{'AUC-PR':>9}{'FPR':>8}")
    for strat, m in results.items():
        print(f"  {strat:<10}{m['f1']:>8}{m['auc_roc']:>10}{m['auc_pr']:>9}{m['false_positive_rate']:>8}")
    print("-" * 60)
    any_strat = next(iter(results.values()))
    print(f"  HEADLINE AUC-ROC: {any_strat['auc_roc']}  (threshold-independent)")
    print(f"  Artefacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
