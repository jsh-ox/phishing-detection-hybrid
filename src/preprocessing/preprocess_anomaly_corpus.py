"""
Pipeline B - Anomaly Detection Corpus Preprocessing
====================================================
Project: Hybrid ML for Phishing Detection (G2)

Transforms the PhiUSIIL URL dataset into a clean, scaled, URL-string-only
feature corpus ready for the anomaly detection layer (Isolation Forest,
feedforward Autoencoder, LSTM Autoencoder).

Key design decisions (documented for methodology):
  - URL-STRING features only. Webpage-content features (LineOfCode, HasFavicon,
    NoOfiFrame, etc.) are EXCLUDED because they are unavailable at joint
    evaluation (G3), where only the raw URL is extracted from an email. Training
    on them would create a train/serve mismatch.
  - URLSimilarityIndex is DROPPED - it is a near-perfect label proxy (leakage).
  - StandardScaler is fit on the TRAIN split ONLY and saved as an artefact,
    then applied unchanged to val/test - prevents data leakage.
  - A legitimate-only training subset is produced for the autoencoders, which
    learn a model of "normal" and flag deviations.

Order of operations:
  1. Load + select URL-string feature subset
  2. Structural cleaning   - nulls, duplicates, zero-variance columns
  3. Stratified split       - 80/10/10, fixed seed
  4. Feature scaling        - StandardScaler fit on train only, saved
  5. Legitimate-only subset - for autoencoder training
  6. Save artefacts + report
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

# URL-string-derived features only, restricted to those that can be
# reproducibly recomputed from a raw URL string at joint evaluation (G3).
# The 3 reference-dependent features (CharContinuationRate, TLDLegitimateProb,
# URLCharProb) are EXCLUDED: they require probability tables learned from the
# PhiUSIIL distribution and cannot be cleanly reproduced for CEAS/TREC email
# URLs, which would create a train/serve feature mismatch.
URL_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfLettersInURL",
    "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL", "NoOfEqualsInURL",
    "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL", "IsHTTPS",
]
LABEL_COL = "label"


def main(input_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "source_file": Path(input_path).name,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "design_notes": {
            "feature_scope": "URL-string-derived features only",
            "excluded": "webpage-content features (unavailable at joint evaluation)",
            "dropped_leak": "URLSimilarityIndex (near-perfect label proxy)",
        },
        "steps": {},
    }

    # --- 1. Load + select ---
    df = pd.read_csv(input_path)
    report["steps"]["loaded_rows"] = len(df)
    # Keep raw URL column through cleaning for correct deduplication, then drop it.
    keep_cols = URL_FEATURES + [LABEL_COL]
    if "URL" in df.columns:
        keep_cols = ["URL"] + keep_cols
    df = df[keep_cols].copy()
    report["steps"]["selected_features"] = len(URL_FEATURES)

    # --- 2. Structural cleaning ---
    n0 = len(df)
    df = df.dropna()
    report["steps"]["removed_null_rows"] = n0 - len(df)

    n1 = len(df)
    # Deduplicate on the raw URL string (genuinely unique per record). Deduping
    # on the coarse 18-feature vector would collapse ~86% of distinct URLs that
    # happen to share the same string-level features - an artefact of removing
    # the high-precision reference features, not true duplication.
    if "URL" in df.columns:
        df = df.drop_duplicates(subset=["URL"])
        df = df.drop(columns=["URL"])
    else:
        df = df.drop_duplicates()
    report["steps"]["removed_duplicate_rows"] = n1 - len(df)
    report["steps"]["dedup_key"] = "raw URL string" if "URL" in pd.read_csv(input_path, nrows=1).columns else "feature vector"

    # zero-variance feature removal
    feature_cols = [c for c in URL_FEATURES]
    variances = df[feature_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        df = df.drop(columns=zero_var)
        feature_cols = [c for c in feature_cols if c not in zero_var]
    report["steps"]["removed_zero_variance"] = zero_var

    # --- 3. Stratified split (80/10/10) ---
    X = df[feature_cols].values
    y = df[LABEL_COL].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED)
    report["steps"]["split_sizes"] = {
        "train": len(X_train), "val": len(X_val), "test": len(X_test)}

    # --- 4. Feature scaling (fit on TRAIN only) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, out / "standard_scaler.pkl")

    # --- 5. Legitimate-only training subset (label 0) for autoencoders ---
    legit_mask = (y_train == 0)
    X_train_legit = X_train_s[legit_mask]
    report["steps"]["legit_only_train_rows"] = int(legit_mask.sum())

    # --- 6. Save artefacts ---
    def save(name, Xarr, yarr=None):
        d = pd.DataFrame(Xarr, columns=feature_cols)
        if yarr is not None:
            d[LABEL_COL] = yarr
        d.to_parquet(out / name, index=False)

    save("url_features_train.parquet", X_train_s, y_train)
    save("url_features_val.parquet", X_val_s, y_val)
    save("url_features_test.parquet", X_test_s, y_test)
    save("url_features_legit_only.parquet", X_train_legit)

    with open(out / "feature_schema.json", "w") as f:
        json.dump({"features": feature_cols, "label": LABEL_COL}, f, indent=2)

    # class distributions
    def dist(y): 
        u, c = np.unique(y, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}
    report["steps"]["class_distribution"] = {
        "train": dist(y_train), "val": dist(y_val), "test": dist(y_test)}

    with open(out / "g2_data_readiness_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- Console summary ---
    print("=" * 60)
    print("PIPELINE B COMPLETE - Anomaly Detection Corpus (G2)")
    print("=" * 60)
    print(f"Source rows loaded     : {report['steps']['loaded_rows']:,}")
    print(f"Features selected      : {report['steps']['selected_features']} (URL-string only)")
    print(f"Removed null rows      : {report['steps']['removed_null_rows']:,}")
    print(f"Removed duplicate rows : {report['steps']['removed_duplicate_rows']:,}")
    print(f"Zero-variance dropped  : {zero_var if zero_var else 'none'}")
    print()
    print("Split sizes:")
    print(f"   Train : {len(X_train):,}")
    print(f"   Val   : {len(X_val):,}")
    print(f"   Test  : {len(X_test):,}")
    print(f"   Legit-only (AE train): {int(legit_mask.sum()):,}")
    print()
    print("Class distribution (train):", dist(y_train))
    print()
    print("Artefacts saved to:", out)
    for p in sorted(out.glob("*")):
        print("   -", p.name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output_dir")
    args = ap.parse_args()
    main(args.input, args.output_dir)
