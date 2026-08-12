#!/usr/bin/env python
# coding: utf-8

# # Anomaly Layer (G5) — Isolation Forest
# 
# Trains and evaluates an Isolation Forest on the G2 URL-feature corpus (18 features).
# 
# **Training regime:** the phishing class is the *majority* (~57%), which violates the Isolation Forest assumption that anomalies are rare. Training on the full set inverts the scoring (it isolates the more-diverse *legitimate* URLs as anomalies). We therefore train on **legitimate URLs only** (novelty detection), so "normal" genuinely means legitimate and phishing becomes the deviation — the same regime used by the autoencoders.
# 
# **Scoring convention:** higher score = more anomalous = more likely phishing.

# In[2]:


import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

# --- Configuration (edit these instead of command-line args) ---
TRAIN_PATH = "url_features_legit_only.parquet"   # legit-only for novelty detection
VAL_PATH   = "url_features_val.parquet"
TEST_PATH  = "url_features_test.parquet"
OUTPUT_DIR = "results/isolation_forest"

LABEL_COL          = "label"
SEED               = 42
N_ESTIMATORS       = 200
MAX_SAMPLES        = "auto"     # 'auto', an int, or a float fraction
CONTAMINATION      = "auto"     # 'auto', 'prior' (full-set only), or a float
LEGIT_ONLY         = True       # train file is legitimate-only, no label column
THRESHOLD_STRATEGY = "f1"       # 'f1', 'youden', or 'fpr'
TARGET_FPR         = 0.10       # used only when THRESHOLD_STRATEGY == 'fpr'


# In[3]:


def load_xy(path):
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c != LABEL_COL]
    X = df[feats].values.astype(float)
    y = df[LABEL_COL].values.astype(int) if LABEL_COL in df.columns else None
    return X, y, feats


def anomaly_scores(model, X):
    """Higher = more anomalous. sklearn score_samples: higher = more normal, so negate."""
    return -model.score_samples(X)


def evaluate(scores, y_true, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, scores), 4),
        "auc_pr": round(average_precision_score(y_true, scores), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
    }


# In[4]:


def pick_threshold(scores, y_true, strategy="f1", target_fpr=0.10):
    """Select an anomaly-score threshold on the validation set.

    'f1'     -> maximise F1
    'youden' -> maximise Youden's J = TPR - FPR (less swayed by class imbalance)
    'fpr'    -> lowest threshold whose FPR <= target_fpr (falls back to youden)
    Returns (threshold, achieved_score_for_the_strategy).
    """
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    best_t, best_val = candidates[0], -1.0

    if strategy == "fpr":
        feasible = []
        for t in candidates:
            pred = (scores >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            if fpr <= target_fpr:
                feasible.append((tpr, t))
        if feasible:
            feasible.sort(reverse=True)   # highest recall meeting the FPR cap
            return float(feasible[0][1]), float(feasible[0][0])
        strategy = "youden"   # fallback if nothing meets the target

    for t in candidates:
        pred = (scores >= t).astype(int)
        if strategy == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            val = tpr - fpr
        else:  # 'f1'
            val = f1_score(y_true, pred, zero_division=0)
        if val > best_val:
            best_val, best_t = val, t
    return float(best_t), float(best_val)


# In[5]:


out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)

X_train, y_train, feats = load_xy(TRAIN_PATH)
X_val,   y_val,   _     = load_xy(VAL_PATH)
X_test,  y_test,  _     = load_xy(TEST_PATH)

print(f"train: {X_train.shape} | val: {X_val.shape} | test: {X_test.shape}")
print(f"features ({len(feats)}): {feats}")


# In[6]:


if LEGIT_ONLY:
    contam = "auto" if CONTAMINATION in ("prior", "auto") else float(CONTAMINATION)
elif CONTAMINATION == "auto":
    contam = "auto"
elif CONTAMINATION == "prior":
    contam = min(float((y_train == 1).mean()), 0.5)   # sklearn caps at 0.5
else:
    contam = float(CONTAMINATION)

ms = MAX_SAMPLES
if ms not in ("auto",):
    ms = float(ms) if "." in str(ms) else int(ms)

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples=ms,
    contamination=contam,
    random_state=SEED,
    n_jobs=-1,
)
model.fit(X_train)
print("Model fitted.")


# In[7]:


val_scores  = anomaly_scores(model, X_val)
threshold, strat_val = pick_threshold(val_scores, y_val,
                                      strategy=THRESHOLD_STRATEGY,
                                      target_fpr=TARGET_FPR)

test_scores = anomaly_scores(model, X_test)
metrics = evaluate(test_scores, y_test, threshold)

print(f"threshold ({THRESHOLD_STRATEGY}) = {threshold:.4f}  (val {THRESHOLD_STRATEGY}={strat_val:.4f})")
print("\nTEST SET METRICS")
for k in ["precision", "recall", "f1", "auc_roc", "auc_pr", "false_positive_rate"]:
    print(f"   {k:<20}: {metrics[k]}")
print(f"   confusion matrix    : {metrics['confusion_matrix']}")


# In[8]:


joblib.dump(model, out / "isolation_forest.pkl")

with open(out / "if_threshold.json", "w") as f:
    json.dump({"threshold": threshold, "strategy": THRESHOLD_STRATEGY,
               "strategy_value": round(strat_val, 4),
               "contamination": contam if contam == "auto" else round(contam, 4)}, f, indent=2)

pd.DataFrame({"anomaly_score": test_scores, "label": y_test}).to_parquet(
    out / "if_scores_test.parquet", index=False)

report = {
    "model": "IsolationForest",
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "n_estimators": N_ESTIMATORS,
    "max_samples": ms,
    "contamination": contam if contam == "auto" else round(contam, 4),
    "n_features": len(feats),
    "train_regime": ("legitimate-only (novelty detection; shared regime with autoencoders)"
                     if LEGIT_ONLY else
                     "full training set (both classes) with contamination"),
    "threshold_strategy": THRESHOLD_STRATEGY,
    "threshold": round(threshold, 6),
    "test_metrics": metrics,
}
with open(out / "if_eval_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"Artefacts saved to: {out}")


# In[9]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# score distributions by class
ax[0].hist(test_scores[y_test == 0], bins=50, alpha=0.6, label="legitimate", density=True)
ax[0].hist(test_scores[y_test == 1], bins=50, alpha=0.6, label="phishing", density=True)
ax[0].axvline(threshold, color="k", ls="--", lw=1, label="threshold")
ax[0].set_title("Anomaly score by class"); ax[0].set_xlabel("anomaly score"); ax[0].legend()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, test_scores)
ax[1].plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']}")
ax[1].plot([0, 1], [0, 1], "k--", lw=1)
ax[1].set_title("ROC curve"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].legend()

plt.tight_layout(); plt.show()


# ## Hyperparameter Sweep — Isolation Forest
# 
# A grid search over `n_estimators` and `max_samples`, training legit-only and evaluating by **AUC-ROC** and **AUC-PR** (threshold-independent, so they measure the model's true discriminative quality rather than an operating-point artefact).
# 
# Assumes the helper functions from the earlier cells are defined: `load_xy`, `anomaly_scores`, `pick_threshold`, `evaluate`, and the `*_PATH` config variables.

# In[10]:


import itertools, time

# --- Sweep grid ---
GRID_N_ESTIMATORS = [100, 200, 300]
GRID_MAX_SAMPLES  = [128, 256, 512]

# Load once (legit-only training; labelled val/test)
X_train, _,      feats = load_xy(TRAIN_PATH)   # legit-only file
X_val,   y_val,  _     = load_xy(VAL_PATH)
X_test,  y_test, _     = load_xy(TEST_PATH)

sweep_rows = []
for n_est, max_s in itertools.product(GRID_N_ESTIMATORS, GRID_MAX_SAMPLES):
    t0 = time.time()
    model = IsolationForest(
        n_estimators=n_est,
        max_samples=max_s,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    ).fit(X_train)

    val_scores  = anomaly_scores(model, X_val)
    test_scores = anomaly_scores(model, X_test)

    # threshold on validation (Youden = balanced, less imbalance-biased than F1)
    thr, _ = pick_threshold(val_scores, y_val, strategy="youden")
    m = evaluate(test_scores, y_test, thr)

    sweep_rows.append({
        "n_estimators": n_est,
        "max_samples":  max_s,
        "auc_roc": m["auc_roc"],
        "auc_pr":  m["auc_pr"],
        "f1":      m["f1"],
        "precision": m["precision"],
        "recall":  m["recall"],
        "fpr":     m["false_positive_rate"],
        "fit_s":   round(time.time() - t0, 1),
    })

sweep = pd.DataFrame(sweep_rows).sort_values("auc_roc", ascending=False).reset_index(drop=True)
sweep


# In[11]:


best = sweep.iloc[0]
print("AUC-ROC range: {:.4f} – {:.4f}  (spread {:.4f})".format(
    sweep.auc_roc.min(), sweep.auc_roc.max(), sweep.auc_roc.max() - sweep.auc_roc.min()))
print("AUC-PR  range: {:.4f} – {:.4f}".format(sweep.auc_pr.min(), sweep.auc_pr.max()))
print()
print("Best configuration by AUC-ROC:")
print(f"  n_estimators = {int(best.n_estimators)}")
print(f"  max_samples  = {int(best.max_samples)}")
print(f"  AUC-ROC = {best.auc_roc} | AUC-PR = {best.auc_pr} | F1 = {best.f1}")


# ### Visualising the sweep
# A heatmap of AUC-ROC across the grid makes the `max_samples` trend obvious. In this setup, **larger `max_samples` improves AUC** — the opposite of the classic Isolation Forest guidance (Liu et al.), because we model the *shape of normal* from a large, diverse legitimate population rather than isolating rare anomalies. Larger subsamples let each tree capture more of that diversity.

# In[12]:


import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

# --- Heatmap of AUC-ROC over the grid ---
pivot = sweep.pivot(index="n_estimators", columns="max_samples", values="auc_roc")
im = ax[0].imshow(pivot.values, cmap="viridis", aspect="auto")
ax[0].set_xticks(range(len(pivot.columns))); ax[0].set_xticklabels(pivot.columns)
ax[0].set_yticks(range(len(pivot.index)));   ax[0].set_yticklabels(pivot.index)
ax[0].set_xlabel("max_samples"); ax[0].set_ylabel("n_estimators")
ax[0].set_title("AUC-ROC across the grid")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        ax[0].text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center",
                   color="white" if pivot.values[i, j] < pivot.values.mean() else "black")
fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

# --- AUC-ROC vs max_samples, one line per n_estimators ---
for n_est in GRID_N_ESTIMATORS:
    sub = sweep[sweep.n_estimators == n_est].sort_values("max_samples")
    ax[1].plot(sub.max_samples, sub.auc_roc, marker="o", label=f"n_estimators={n_est}")
ax[1].axhline(0.5, color="grey", ls="--", lw=1, label="chance (0.5)")
ax[1].set_xlabel("max_samples"); ax[1].set_ylabel("AUC-ROC")
ax[1].set_title("AUC-ROC vs max_samples"); ax[1].legend()

plt.tight_layout(); plt.show()


# In[13]:


from pathlib import Path
out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
sweep.to_csv(out / "if_sweep_results.csv", index=False)
print(f"Sweep results saved to {out / 'if_sweep_results.csv'}")


# In[ ]:




