#!/usr/bin/env python
# coding: utf-8

# # Anomaly Layer (G5) — LSTM Autoencoder
# 
# Consumes **raw URL character sequences**, not the 18 engineered features, leveraging its sequential analysis giving the LSTM a distinct mechanism (sequence reconstruction) from the feedforward autoencoder (feature reconstruction) and Isolation Forest (feature-space partitioning).
# 
# **Legit-only training**, identical splits and seed, threshold on validation, evaluated by AUC-ROC / AUC-PR.
# 
# Pre-processing pipeline: **Run `prepare_url_sequences.py`** to produce the `.npy` artefacts.

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

DATA_DIR   = "lstm_data"                  # output of prepare_url_sequences.py
OUTPUT_DIR = "results/autoencoder_lstm"

SEED          = 42
EMBED_DIM     = 32      # character embedding size
LATENT_DIM    = 64      # LSTM hidden state = the bottleneck
EPOCHS        = 30
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
VAL_SPLIT_FROM_TRAIN = 0.1
THRESHOLD_STRATEGY   = "youden"
TARGET_FPR    = 0.10

tf.random.set_seed(SEED); np.random.seed(SEED)


# In[2]:


d = Path(DATA_DIR)
X_train = np.load(d / "url_seq_train_legit.npy")     # legit-only
X_val   = np.load(d / "url_seq_val.npy")
X_test  = np.load(d / "url_seq_test.npy")
y_val   = np.load(d / "url_seq_val_labels.npy")
y_test  = np.load(d / "url_seq_test_labels.npy")

meta = json.load(open(d / "char_vocab.json"))
VOCAB_SIZE = meta["vocab_size"]
MAX_LEN    = meta["max_len"]

print(f"train (legit-only): {X_train.shape}")
print(f"val: {X_val.shape} | phishing frac {y_val.mean():.3f}")
print(f"test: {X_test.shape} | phishing frac {y_test.mean():.3f}")
print(f"vocab size: {VOCAB_SIZE} | sequence length: {MAX_LEN}")


# In[3]:


def build_lstm_autoencoder(vocab_size, max_len, embed_dim, latent_dim, lr):
    """Sequence-to-sequence autoencoder over URL characters.

    Encoder: chars -> embedding -> LSTM -> latent vector (the bottleneck)
    Decoder: latent repeated max_len times -> LSTM -> per-position char distribution

    Reconstruction target is the input character sequence itself, so we use
    sparse categorical crossentropy: the model must re-predict each character.
    """
    inp = keras.Input(shape=(max_len,), dtype="int32", name="url_chars")

    # --- Encoder ---
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="char_embed")(inp)
    latent = layers.LSTM(latent_dim, name="encoder_lstm")(x)     # final state = bottleneck

    # --- Decoder ---
    x = layers.RepeatVector(max_len, name="repeat_latent")(latent)
    x = layers.LSTM(latent_dim, return_sequences=True, name="decoder_lstm")(x)
    out = layers.TimeDistributed(
        layers.Dense(vocab_size, activation="softmax"), name="char_out")(x)

    model = keras.Model(inp, out, name="lstm_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
    )
    return model

lstm_ae = build_lstm_autoencoder(VOCAB_SIZE, MAX_LEN, EMBED_DIM, LATENT_DIM, LEARNING_RATE)
lstm_ae.summary()


# In[4]:


early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True)

history = lstm_ae.fit(
    X_train, X_train,                 # target = input sequence (autoencoder)
    validation_split=VAL_SPLIT_FROM_TRAIN,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=2,
)
print(f"Stopped at epoch {len(history.history['loss'])}, "
      f"best val_loss {min(history.history['val_loss']):.5f}")


# In[5]:


def sequence_reconstruction_error(model, X, batch_size=512):
    """Per-sample mean cross-entropy of reconstructing the character sequence.
    Higher = the model struggled to rebuild this URL = more anomalous.
    Padding positions are masked out so they don't dilute the score.
    """
    errors = np.zeros(len(X), dtype="float32")
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        probs = model.predict(batch, verbose=0)                  # (b, max_len, vocab)
        # probability the model assigned to the TRUE character at each position
        idx = np.take_along_axis(probs, batch[..., None], axis=-1).squeeze(-1)
        ce = -np.log(np.clip(idx, 1e-9, 1.0))                    # per-position CE
        mask = (batch != 0).astype("float32")                    # ignore padding
        errors[i:i + batch_size] = (ce * mask).sum(1) / np.maximum(mask.sum(1), 1)
    return errors

val_scores  = sequence_reconstruction_error(lstm_ae, X_val)
test_scores = sequence_reconstruction_error(lstm_ae, X_test)

print("Mean reconstruction error (val):")
print(f"  legitimate: {val_scores[y_val == 0].mean():.5f}")
print(f"  phishing  : {val_scores[y_val == 1].mean():.5f}")
print(f"  separation: {val_scores[y_val == 1].mean() - val_scores[y_val == 0].mean():+.5f}")


# In[6]:


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

threshold, strat_val = pick_threshold(val_scores, y_val, THRESHOLD_STRATEGY, TARGET_FPR)
metrics = evaluate(test_scores, y_test, threshold)

print(f"threshold ({THRESHOLD_STRATEGY}) = {threshold:.6f}")
print("\nTEST SET METRICS")
for k in ["precision", "recall", "f1", "auc_roc", "auc_pr", "false_positive_rate"]:
    print(f"   {k:<20}: {metrics[k]}")
print(f"   confusion matrix    : {metrics['confusion_matrix']}")


# In[7]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(1, 3, figsize=(16, 4))

ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="val")
ax[0].set_title("LSTM AE training"); ax[0].set_xlabel("epoch")
ax[0].set_ylabel("cross-entropy"); ax[0].legend()

ax[1].hist(test_scores[y_test == 0], bins=60, alpha=0.6, label="legitimate", density=True)
ax[1].hist(test_scores[y_test == 1], bins=60, alpha=0.6, label="phishing", density=True)
ax[1].axvline(threshold, color="k", ls="--", lw=1, label="threshold")
ax[1].set_title("Reconstruction error by class"); ax[1].set_xlabel("mean CE"); ax[1].legend()

fpr, tpr, _ = roc_curve(y_test, test_scores)
ax[2].plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']}")
ax[2].plot([0, 1], "k--", lw=1)
ax[2].set_title("ROC curve"); ax[2].set_xlabel("FPR"); ax[2].set_ylabel("TPR"); ax[2].legend()

plt.tight_layout(); plt.show()


# In[ ]:


out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)

lstm_ae.save(out / "autoencoder_lstm.keras")
pd.DataFrame({"anomaly_score": test_scores, "label": y_test}).to_parquet(
    out / "ae_lstm_scores_test.parquet", index=False)

report = {
    "model": "LSTMAutoencoder",
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "input_representation": "raw URL character sequences (not the 18 engineered features)",
    "architecture": {"vocab_size": VOCAB_SIZE, "max_len": MAX_LEN,
                     "embed_dim": EMBED_DIM, "latent_dim": LATENT_DIM},
    "training": {"epochs_run": len(history.history["loss"]),
                 "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE},
    "train_regime": "legitimate-only (sequence reconstruction novelty detection)",
    "threshold_strategy": THRESHOLD_STRATEGY,
    "threshold": round(float(threshold), 6),
    "test_metrics": metrics,
}
with open(out / "ae_lstm_eval_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"Artefacts saved to {out}")

