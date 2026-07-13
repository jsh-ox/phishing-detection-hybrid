"""
LSTM Autoencoder — Character-Sequence Data Preparation
=======================================================
Project: Hybrid ML for Phishing Detection (G5)

Produces character-index sequences using the same splits and legit-only
training regime as the other anomaly models:
  - dedup on raw URL string (matching the G2 corpus)
  - stratified 80/10/10 with the same seed
  - legit-only training subset (label == 0) for the autoencoder

Outputs:
  url_seq_train_legit.npy   int32 (n, MAX_LEN)  legitimate only, for training
  url_seq_val.npy  + labels
  url_seq_test.npy + labels
  char_vocab.json           char -> index mapping
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
MAX_LEN = 128
PAD_IDX = 0
OOV_IDX = 1


def build_vocab(urls):
    """Character vocabulary from the TRAINING urls only"""
    chars = sorted({c for u in urls for c in str(u)})
    # 0 = pad, 1 = OOV, real chars start at 2
    return {c: i + 2 for i, c in enumerate(chars)}


def encode(urls, vocab, max_len=MAX_LEN):
    """Char -> index, truncated/padded (post) to max_len."""
    out = np.full((len(urls), max_len), PAD_IDX, dtype=np.int32)
    for i, u in enumerate(urls):
        for j, c in enumerate(str(u)[:max_len]):
            out[i, j] = vocab.get(c, OOV_IDX)
    return out


def main(source_csv, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv, usecols=["URL", "label"])
    df = df.drop_duplicates(subset=["URL"]).reset_index(drop=True)   # same key as G2

    # Stratified 80/10/10 split
    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)

    # Vocabulary from TRAINING urls only
    vocab = build_vocab(train_df["URL"].tolist())

    # Legit-only training subset (label == 0)
    train_legit = train_df[train_df["label"] == 0]

    X_train_legit = encode(train_legit["URL"].tolist(), vocab)
    X_val = encode(val_df["URL"].tolist(), vocab)
    X_test = encode(test_df["URL"].tolist(), vocab)

    np.save(out / "url_seq_train_legit.npy", X_train_legit)
    np.save(out / "url_seq_val.npy", X_val)
    np.save(out / "url_seq_test.npy", X_test)
    np.save(out / "url_seq_val_labels.npy", val_df["label"].values.astype(np.int32))
    np.save(out / "url_seq_test_labels.npy", test_df["label"].values.astype(np.int32))

    with open(out / "char_vocab.json", "w") as f:
        json.dump({"vocab": vocab, "max_len": MAX_LEN,
                   "pad_idx": PAD_IDX, "oov_idx": OOV_IDX,
                   "vocab_size": len(vocab) + 2}, f, indent=2)

    print("=" * 56)
    print("LSTM CHARACTER-SEQUENCE PREPARED DATA")
    print("=" * 56)
    print(f"Source rows (deduped) : {len(df):,}")
    print(f"Split  train/val/test : {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
    print(f"Legit-only training   : {len(train_legit):,}")
    print(f"Vocabulary size       : {len(vocab) + 2} (incl. pad + OOV)")
    print(f"Sequence length       : {MAX_LEN}")
    print(f"Val  phishing frac    : {val_df['label'].mean():.3f}")
    print(f"Test phishing frac    : {test_df['label'].mean():.3f}")
    print(f"Artefacts saved to    : {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("source_csv")
    ap.add_argument("output_dir")
    args = ap.parse_args()
    main(args.source_csv, args.output_dir)
