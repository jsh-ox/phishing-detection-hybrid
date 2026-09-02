"""
Anomaly Layer — Feedforward Autoencoder
=====================================================================================
Project: Hybrid ML for Phishing Detection

A dense autoencoder trained on LEGITIMATE-ONLY URLs. It learns to reconstruct
normal (legitimate) feature vectors; phishing URLs — unseen in training —
reconstruct poorly, so per-sample reconstruction error (MSE) is the anomaly
score

Outputs (into OUTPUT_DIR):
  autoencoder_ff.keras        trained model
  ae_ff_scores_test.parquet   per-row test anomaly scores + label (for fusion)
  ae_ff_eval_report.json      metrics under all three threshold strategies
  training_curve.png          train/val reconstruction loss
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from anomaly_eval import to_phishing_target, evaluate_all_strategies

# --- CONFIG ---
DATA_DIR   = Path("data") / "processed" / "G2_v2_nohttps"
TRAIN_PATH = DATA_DIR / "url_features_legit_only.parquet"
VAL_PATH   = DATA_DIR / "url_features_val.parquet"
TEST_PATH  = DATA_DIR / "url_features_test.parquet"
OUTPUT_DIR = Path("results/anomaly/autoencoder_ff_nohttps")

LABEL_COL      = "label"
SEED           = 42
ENCODING_DIM   = 8
HIDDEN_DIMS    = [16]
DROPOUT        = 0.0

EPOCHS         = 100
BATCH_SIZE     = 256
LEARNING_RATE  = 1e-3
VAL_SPLIT      = 0.1
PATIENCE       = 10
TARGET_FPR     = 0.10
# ======================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_features(path):
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c != LABEL_COL]
    X = df[feats].values.astype("float32")
    y = df[LABEL_COL].values if LABEL_COL in df.columns else None
    return X, feats, y


def build_autoencoder(n_features, encoding_dim, hidden_dims, dropout, lr):
    inp = keras.Input(shape=(n_features,), name="input")
    x = inp
    for h in hidden_dims:
        x = layers.Dense(h, activation="relu")(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    x = layers.Dense(encoding_dim, activation="relu", name="bottleneck")(x)
    for h in reversed(hidden_dims):
        x = layers.Dense(h, activation="relu")(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_features, activation="linear", name="reconstruction")(x)

    model = keras.Model(inp, out, name="feedforward_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model


def reconstruction_error(model, X):
    """Per-sample MSE between input and reconstruction. Higher = more anomalous."""
    recon = model.predict(X, batch_size=512, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, feats, _ = load_features(TRAIN_PATH)
    X_val, _, y_val_raw  = load_features(VAL_PATH)
    X_test, _, y_test_raw = load_features(TEST_PATH)
    n_features = len(feats)

    y_val = to_phishing_target(y_val_raw)
    y_test = to_phishing_target(y_test_raw)

    model = build_autoencoder(n_features, ENCODING_DIM, HIDDEN_DIMS, DROPOUT, LEARNING_RATE)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        X_train, X_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=2,
    )

    val_scores = reconstruction_error(model, X_val)
    test_scores = reconstruction_error(model, X_test)

    # Phishing SHOULD reconstruct worse than legit
    sep_legit = float(val_scores[y_val == 0].mean())   # y_val==0 -> legitimate
    sep_phish = float(val_scores[y_val == 1].mean())   # y_val==1 -> phishing

    results = evaluate_all_strategies(val_scores, y_val, test_scores, y_test, TARGET_FPR)

    model.save(OUTPUT_DIR / "autoencoder_ff.keras")
    pd.DataFrame({"anomaly_score": test_scores, "label": y_test_raw}).to_parquet(
        OUTPUT_DIR / "ae_ff_scores_test.parquet", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.xlabel("epoch"); plt.ylabel("MSE reconstruction loss")
        plt.title("Feedforward AE training"); plt.legend(); plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "training_curve.png", dpi=120)
        plt.close()
    except Exception as e:
        print("(training curve skipped:", e, ")")

    report = {
        "model": "FeedforwardAutoencoder",
        "corpus": "G2_v2",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "label_convention": "0=phishing, 1=legitimate; detection target = phishing",
        "train_regime": "legitimate-only",
        "n_features": n_features,
        "architecture": {"encoding_dim": ENCODING_DIM, "hidden_dims": HIDDEN_DIMS,
                         "dropout": DROPOUT},
        "training": {"epochs_run": len(history.history["loss"]),
                     "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE},
        "train_rows": len(X_train),
        "reconstruction_error_separation": {
            "legit_mean": round(sep_legit, 5),
            "phish_mean": round(sep_phish, 5),
            "separation": round(sep_phish - sep_legit, 5),
        },
        "test_metrics_by_strategy": results,
    }
    with open(OUTPUT_DIR / "ae_ff_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # summary
    print("=" * 60)
    print("FEEDFORWARD AUTOENCODER — G2 v2")
    print("=" * 60)
    print(f"  features         : {n_features}")
    print(f"  legit-only train : {len(X_train):,}")
    print(f"  bottleneck       : {ENCODING_DIM} | hidden: {HIDDEN_DIMS} | dropout: {DROPOUT}")
    print(f"  epochs run       : {len(history.history['loss'])}")
    print("-" * 60)
    print(f"  recon error  legit={sep_legit:.5f}  phish={sep_phish:.5f}  "
          f"separation={sep_phish - sep_legit:+.5f}")
    print("  (separation should be POSITIVE: phishing reconstructs worse)")
    print("-" * 60)
    print(f"  {'strategy':<10}{'F1':>8}{'AUC-ROC':>10}{'AUC-PR':>9}{'FPR':>8}")
    for strat, m in results.items():
        print(f"  {strat:<10}{m['f1']:>8}{m['auc_roc']:>10}{m['auc_pr']:>9}{m['false_positive_rate']:>8}")
    print("-" * 60)
    print(f"  HEADLINE AUC-ROC: {next(iter(results.values()))['auc_roc']}  (threshold-independent)")
    print(f"  Artefacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
