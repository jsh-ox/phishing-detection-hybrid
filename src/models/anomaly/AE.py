#!/usr/bin/env python
# coding: utf-8

# # Anomaly Layer (G5) — Feedforward Autoencoder
# 
# A feedforward (dense) autoencoder trained on **legitimate URLs only**. It learns to reconstruct normal feature vectors such that phishing URLs — unseen in training — reconstruct poorly. **Reconstruction error is the anomaly score** (higher = more anomalous = more likely phishing).
# 
# Legit-only training, threshold chosen on the validation set, evaluated by AUC-ROC / AUC-PR (threshold-independent) plus precision/recall/F1 at the chosen operating point.

# In[1]:


import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

# --- Configuration ---
TRAIN_PATH = "url_features_legit_only.parquet"   # legit-only, no label column
VAL_PATH   = "url_features_val.parquet"
TEST_PATH  = "url_features_test.parquet"
OUTPUT_DIR = "results/autoencoder_ff"

LABEL_COL          = "label"
SEED               = 42
# --- Architecture ---
ENCODING_DIM       = 8       # bottleneck size (18 features -> 8 -> 18)
HIDDEN_DIMS        = [14]    # optional intermediate layer(s) each side of the bottleneck
DROPOUT            = 0.0     # 0.0 = off; small values (0.1) can regularise
# --- Training ---
EPOCHS             = 100
BATCH_SIZE         = 256
LEARNING_RATE      = 1e-3
VAL_SPLIT_FROM_TRAIN = 0.1   # carve a validation slice from the legit training data
                             # for early stopping (kept separate from the labelled VAL set)
THRESHOLD_STRATEGY = "youden"  # 'f1', 'youden', or 'fpr'
TARGET_FPR         = 0.10

# Reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)


# In[2]:


def load_xy(path):
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c != LABEL_COL]
    X = df[feats].values.astype("float32")
    y = df[LABEL_COL].values.astype(int) if LABEL_COL in df.columns else None
    return X, y, feats

X_train, _,      feats = load_xy(TRAIN_PATH)   # legit-only
X_val,   y_val,  _     = load_xy(VAL_PATH)
X_test,  y_test, _     = load_xy(TEST_PATH)
N_FEATURES = len(feats)

print(f"train (legit-only): {X_train.shape}")
print(f"val:  {X_val.shape} | phishing frac {y_val.mean():.3f}")
print(f"test: {X_test.shape} | phishing frac {y_test.mean():.3f}")
print(f"features ({N_FEATURES}): {feats}")


# In[3]:


def build_autoencoder(n_features, encoding_dim, hidden_dims, dropout, lr):
    inp = keras.Input(shape=(n_features,), name="input")
    x = inp
    # Encoder
    for h in hidden_dims:
        x = layers.Dense(h, activation="relu")(x)
        if dropout: x = layers.Dropout(dropout)(x)
    bottleneck = layers.Dense(encoding_dim, activation="relu", name="bottleneck")(x)
    # Decoder (mirror)
    x = bottleneck
    for h in reversed(hidden_dims):
        x = layers.Dense(h, activation="relu")(x)
        if dropout: x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_features, activation="linear", name="reconstruction")(x)

    model = keras.Model(inp, out, name="feedforward_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model

autoencoder = build_autoencoder(N_FEATURES, ENCODING_DIM, HIDDEN_DIMS, DROPOUT, LEARNING_RATE)
autoencoder.summary()


# In[4]:


early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True)

history = autoencoder.fit(
    X_train, X_train,                 # target IS the input — that's the autoencoder
    validation_split=VAL_SPLIT_FROM_TRAIN,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=2,
)
print(f"Stopped at epoch {len(history.history['loss'])}, "
      f"best val_loss {min(history.history['val_loss']):.5f}")


# In[5]:


import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("epoch"); plt.ylabel("MSE reconstruction loss")
plt.title("Autoencoder training"); plt.legend(); plt.tight_layout(); plt.show()


# In[6]:


def reconstruction_error(model, X):
    """Per-sample MSE between input and reconstruction. Higher = more anomalous."""
    recon = model.predict(X, batch_size=512, verbose=0)
    return np.mean(np.square(X - recon), axis=1)

val_scores  = reconstruction_error(autoencoder, X_val)
test_scores = reconstruction_error(autoencoder, X_test)

# Quick check: phishing should, on average, reconstruct worse than legitimate
print("Mean reconstruction error:")
print(f"  legitimate (val): {val_scores[y_val == 0].mean():.5f}")
print(f"  phishing   (val): {val_scores[y_val == 1].mean():.5f}")


# In[7]:


def pick_threshold(scores, y_true, strategy="youden", target_fpr=0.10):
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    best_t, best_val = candidates[0], -1.0
    if strategy == "fpr":
        feasible = []
        for t in candidates:
            pred = (scores >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            if fpr <= target_fpr: feasible.append((tpr, t))
        if feasible:
            feasible.sort(reverse=True)
            return float(feasible[0][1]), float(feasible[0][0])
        strategy = "youden"
    for t in candidates:
        pred = (scores >= t).astype(int)
        if strategy == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            val = tpr - fpr
        else:
            val = f1_score(y_true, pred, zero_division=0)
        if val > best_val: best_val, best_t = val, t
    return float(best_t), float(best_val)

threshold, strat_val = pick_threshold(val_scores, y_val, THRESHOLD_STRATEGY, TARGET_FPR)
print(f"threshold ({THRESHOLD_STRATEGY}) = {threshold:.6f}  (val {THRESHOLD_STRATEGY}={strat_val:.4f})")


# In[8]:


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

metrics = evaluate(test_scores, y_test, threshold)
print("TEST SET METRICS")
for k in ["precision", "recall", "f1", "auc_roc", "auc_pr", "false_positive_rate"]:
    print(f"   {k:<20}: {metrics[k]}")
print(f"   confusion matrix    : {metrics['confusion_matrix']}")


# In[9]:


from sklearn.metrics import roc_curve
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(test_scores[y_test == 0], bins=60, alpha=0.6, label="legitimate", density=True)
ax[0].hist(test_scores[y_test == 1], bins=60, alpha=0.6, label="phishing", density=True)
ax[0].axvline(threshold, color="k", ls="--", lw=1, label="threshold")
ax[0].set_xlim(0, np.quantile(test_scores, 0.99))  # clip long tail for readability
ax[0].set_title("Reconstruction error by class"); ax[0].set_xlabel("MSE"); ax[0].legend()

fpr, tpr, _ = roc_curve(y_test, test_scores)
ax[1].plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']}")
ax[1].plot([0, 1], [0, 1], "k--", lw=1)
ax[1].set_title("ROC curve"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].legend()
plt.tight_layout(); plt.show()


# In[10]:


out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)

autoencoder.save(out / "autoencoder_ff.keras")
pd.DataFrame({"anomaly_score": test_scores, "label": y_test}).to_parquet(
    out / "ae_ff_scores_test.parquet", index=False)

report = {
    "model": "FeedforwardAutoencoder",
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "architecture": {"n_features": N_FEATURES, "encoding_dim": ENCODING_DIM,
                     "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT},
    "training": {"epochs_run": len(history.history["loss"]), "batch_size": BATCH_SIZE,
                 "learning_rate": LEARNING_RATE},
    "train_regime": "legitimate-only (reconstruction-based novelty detection)",
    "threshold_strategy": THRESHOLD_STRATEGY,
    "threshold": round(float(threshold), 6),
    "test_metrics": metrics,
}
with open(out / "ae_ff_eval_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"Artefacts saved to {out}")


# ## Bottleneck Sweep — Feedforward Autoencoder
# 
# Grid search over the **bottleneck dimension** (`ENCODING_DIM`) and **dropout**, the autoencoder's highest-leverage architectural levers. Each model trains legit-only; we evaluate by **AUC-ROC** and **AUC-PR** (threshold-independent) plus F1 at a Youden-selected operating point.
# 
# The bottleneck controls how hard the network must compress: too large and it can nearly copy the input (learning little that separates normal from anomalous); too small and it loses legitimate structure too. We're looking for the compression that

# In[11]:


import itertools, time

# --- Sweep grid ---
GRID_ENCODING_DIM = [3, 5, 8, 10, 12]
GRID_DROPOUT      = [0.0, 0.1]
SWEEP_HIDDEN_DIMS = [14]      # keep architecture depth fixed; vary bottleneck only
SWEEP_EPOCHS      = 100       # early stopping will usually cut this short

ae_sweep_rows = []
for enc_dim, dropout in itertools.product(GRID_ENCODING_DIM, GRID_DROPOUT):
    t0 = time.time()

    # fresh seeds each run so results are comparable and reproducible
    tf.random.set_seed(SEED); np.random.seed(SEED)

    model = build_autoencoder(N_FEATURES, enc_dim, SWEEP_HIDDEN_DIMS, dropout, LEARNING_RATE)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                       restore_best_weights=True)
    hist = model.fit(
        X_train, X_train,
        validation_split=VAL_SPLIT_FROM_TRAIN,
        epochs=SWEEP_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[es], verbose=0,
    )

    val_scores  = reconstruction_error(model, X_val)
    test_scores = reconstruction_error(model, X_test)
    thr, _ = pick_threshold(val_scores, y_val, strategy="youden")
    m = evaluate(test_scores, y_test, thr)

    ae_sweep_rows.append({
        "encoding_dim": enc_dim,
        "dropout": dropout,
        "auc_roc": m["auc_roc"],
        "auc_pr": m["auc_pr"],
        "f1": m["f1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "fpr": m["false_positive_rate"],
        "epochs_run": len(hist.history["loss"]),
        "fit_s": round(time.time() - t0, 1),
        # diagnostic: do phishing URLs really reconstruct worse than legitimate?
        "err_legit": round(float(val_scores[y_val == 0].mean()), 5),
        "err_phish": round(float(val_scores[y_val == 1].mean()), 5),
    })
    print(f"enc_dim={enc_dim:>2} dropout={dropout}: "
          f"AUC-ROC={m['auc_roc']:.4f} AUC-PR={m['auc_pr']:.4f} "
          f"({len(hist.history['loss'])} epochs, {time.time()-t0:.1f}s)")

ae_sweep = (pd.DataFrame(ae_sweep_rows)
            .sort_values("auc_roc", ascending=False)
            .reset_index(drop=True))
ae_sweep


# In[12]:


best = ae_sweep.iloc[0]
baseline = 0.6639   # AUC-ROC from the enc_dim=8, dropout=0 baseline run

print("AUC-ROC range: {:.4f} – {:.4f}  (spread {:.4f})".format(
    ae_sweep.auc_roc.min(), ae_sweep.auc_roc.max(),
    ae_sweep.auc_roc.max() - ae_sweep.auc_roc.min()))
print("AUC-PR  range: {:.4f} – {:.4f}".format(ae_sweep.auc_pr.min(), ae_sweep.auc_pr.max()))
print()
print("Best configuration by AUC-ROC:")
print(f"  encoding_dim = {int(best.encoding_dim)} | dropout = {best.dropout}")
print(f"  AUC-ROC = {best.auc_roc} | AUC-PR = {best.auc_pr} | F1 = {best.f1}")
print(f"  vs baseline (enc_dim=8): {best.auc_roc - baseline:+.4f} AUC-ROC")
print()
# sanity: reconstruction error should be higher for phishing across configs
print("Reconstruction-error separation (phish - legit), by config:")
sep = ae_sweep.assign(separation=(ae_sweep.err_phish - ae_sweep.err_legit).round(5))
print(sep[["encoding_dim", "dropout", "separation", "auc_roc"]].to_string(index=False))


# ### Visualising the bottleneck sweep
# AUC-ROC as a function of bottleneck size shows whether there's a compression sweet spot. A peak at a smaller `encoding_dim` would indicate the model separates best when forced to compress harder; a flat line would indicate the feature set — not the architecture — is the limit.

# In[13]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

# AUC-ROC vs encoding_dim, one line per dropout
for dropout in GRID_DROPOUT:
    sub = ae_sweep[ae_sweep.dropout == dropout].sort_values("encoding_dim")
    ax[0].plot(sub.encoding_dim, sub.auc_roc, marker="o", label=f"dropout={dropout}")
ax[0].axhline(0.5, color="grey", ls="--", lw=1, label="chance (0.5)")
ax[0].axhline(baseline, color="red", ls=":", lw=1, label="baseline (enc=8)")
ax[0].set_xlabel("encoding_dim (bottleneck)"); ax[0].set_ylabel("AUC-ROC")
ax[0].set_title("AUC-ROC vs bottleneck size"); ax[0].legend()

# AUC-ROC vs AUC-PR scatter, annotated with encoding_dim
for _, r in ae_sweep.iterrows():
    ax[1].scatter(r.auc_roc, r.auc_pr, s=40)
    ax[1].annotate(f"{int(r.encoding_dim)}", (r.auc_roc, r.auc_pr),
                   textcoords="offset points", xytext=(4, 4), fontsize=8)
ax[1].set_xlabel("AUC-ROC"); ax[1].set_ylabel("AUC-PR")
ax[1].set_title("AUC-ROC vs AUC-PR (labelled by encoding_dim)")

plt.tight_layout(); plt.show()


# In[14]:


from pathlib import Path
out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
ae_sweep.to_csv(out / "ae_ff_bottleneck_sweep.csv", index=False)
print(f"Sweep results saved to {out / 'ae_ff_bottleneck_sweep.csv'}")


# In[ ]:




