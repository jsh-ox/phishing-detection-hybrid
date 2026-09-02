"""
Anomaly Layer — LSTM Autoencoder (G2 v2)
==========================================================
Project: Hybrid ML for Phishing Detection

Unlike the other two anomaly models, this consumes RAW URL CHARACTER SEQUENCES
(not the 24 engineered features).

Trained on LEGITIMATE-ONLY URL sequences (label==1). Reconstruction difficulty
(per-character cross-entropy) is the anomaly score: phishing URLs, unseen in
training, reconstruct worse

Run prepare_url_sequences.py first to produce the .npy artefacts.

Outputs (into OUTPUT_DIR):
  autoencoder_lstm.keras       trained model
  ae_lstm_scores_test.parquet  per-row test anomaly scores + label (for fusion)
  ae_lstm_eval_report.json     metrics under all three threshold strategies
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
DATA_DIR   = Path("data") / "processed" / "G2_v2" / "lstm_data"
OUTPUT_DIR = Path("results/anomaly/autoencoder_lstm")

SEED          = 42
EMBED_DIM     = 32
LATENT_DIM    = 64
EPOCHS        = 30
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.1
PATIENCE      = 5
TARGET_FPR    = 0.10

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_lstm_autoencoder(vocab_size, max_len, embed_dim, latent_dim, lr):
    """Sequence-to-sequence autoencoder over URL characters.
    Encoder: chars -> embedding -> LSTM -> latent (bottleneck)
    Decoder: latent repeated max_len -> LSTM -> per-position char distribution
    """
    inp = keras.Input(shape=(max_len,), dtype="int32", name="url_chars")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="char_embed")(inp)
    latent = layers.LSTM(latent_dim, name="encoder_lstm")(x)
    x = layers.RepeatVector(max_len, name="repeat_latent")(latent)
    x = layers.LSTM(latent_dim, return_sequences=True, name="decoder_lstm")(x)
    out = layers.TimeDistributed(
        layers.Dense(vocab_size, activation="softmax"), name="char_out")(x)

    model = keras.Model(inp, out, name="lstm_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy")
    return model


def sequence_reconstruction_error(model, X, batch_size=512):
    """Per-sample mean cross-entropy of reconstructing the sequence.
    Padding positions are masked out. Higher = harder to rebuild = more anomalous.
    """
    errors = np.zeros(len(X), dtype="float32")
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        probs = model.predict(batch, verbose=0)
        idx = np.take_along_axis(probs, batch[..., None], axis=-1).squeeze(-1)
        ce = -np.log(np.clip(idx, 1e-9, 1.0))
        mask = (batch != 0).astype("float32")
        errors[i:i + batch_size] = (ce * mask).sum(1) / np.maximum(mask.sum(1), 1)
    return errors


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train = np.load(DATA_DIR / "seq_train_legit.npy")
    X_val   = np.load(DATA_DIR / "seq_val.npy")
    X_test  = np.load(DATA_DIR / "seq_test.npy")
    y_val_raw  = np.load(DATA_DIR / "val_labels.npy")
    y_test_raw = np.load(DATA_DIR / "test_labels.npy")

    meta = json.load(open(DATA_DIR / "char_vocab.json"))
    VOCAB_SIZE, MAX_LEN = meta["vocab_size"], meta["max_len"]

    y_val  = to_phishing_target(y_val_raw)
    y_test = to_phishing_target(y_test_raw)

    model = build_lstm_autoencoder(VOCAB_SIZE, MAX_LEN, EMBED_DIM, LATENT_DIM, LEARNING_RATE)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        X_train, X_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=2,
    )

    val_scores  = sequence_reconstruction_error(model, X_val)
    test_scores = sequence_reconstruction_error(model, X_test)

    sep_legit = float(val_scores[y_val == 0].mean())    # legitimate
    sep_phish = float(val_scores[y_val == 1].mean())    # phishing

    results = evaluate_all_strategies(val_scores, y_val, test_scores, y_test, TARGET_FPR)

    model.save(OUTPUT_DIR / "autoencoder_lstm.keras")
    pd.DataFrame({"anomaly_score": test_scores, "label": y_test_raw}).to_parquet(
        OUTPUT_DIR / "ae_lstm_scores_test.parquet", index=False)

    report = {
        "model": "LSTMAutoencoder",
        "corpus": "G2_v2 raw URLs",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "label_convention": "0=phishing, 1=legitimate; detection target = phishing",
        "input_representation": "raw URL character sequences",
        "train_regime": "legitimate-only",
        "architecture": {"vocab_size": VOCAB_SIZE, "max_len": MAX_LEN,
                         "embed_dim": EMBED_DIM, "latent_dim": LATENT_DIM},
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
    with open(OUTPUT_DIR / "ae_lstm_eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("LSTM AUTOENCODER — G2 v2")
    print("=" * 60)
    print(f"  vocab size       : {VOCAB_SIZE} | max_len: {MAX_LEN}")
    print(f"  legit-only train : {len(X_train):,}")
    print(f"  latent (bottleneck): {LATENT_DIM} | embed: {EMBED_DIM}")
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
