"""
G2 Anomaly Corpus Preprocessing — End-to-End (v2, corrected)
============================================================
Project: Hybrid ML for Phishing Detection

Single reproducible pipeline: three raw URL datasets -> combined, label-aligned
corpus -> enriched URL features -> scaled, stratified train/val/test splits +
legit-only subset for novelty-detection training.

LABEL CONVENTION (publisher-confirmed, verified across all sources):
    0 = phishing, 1 = legitimate

This corrects a prior inversion in which label==0 was mistakenly treated as
legitimate; the anomaly models were consequently trained on phishing URLs as
"normal". Here the legit-only subset is label==1 (genuinely legitimate).

Sources & their native conventions (verified against the confirmed reference):
  - PhiUSIIL (Prasad) : columns URL,label      -> 0=phish, 1=legit (as-is)
  - Jushnu            : columns url,type        -> strings mapped phishing->0, legitimate->1
  - Rachana           : columns URL,ClassLabel  -> 0=phish, 1=legit (verified as-is; NOT flipped)

Stages:
  1. Load each source, normalise to (url, label, source) with unified convention
  2. Concatenate; drop URLs with conflicting labels across sources; dedup on raw URL
  3. Extract enriched URL-string features (shared extractor module)
  4. Stratified 80/10/10 split (seed 42)
  5. StandardScaler fit on TRAIN only; transform all splits
  6. Carve legit-only (label==1) training subset for autoencoders / Isolation Forest
  7. Save all artefacts + a readiness report

Usage:
  # Simplest — reads Prasad_v2.csv, Jushnu_K.csv, Rachana_P.csv from data/raw:
  python preprocess_anomaly_corpus.py

  # Custom locations / filenames:
  python preprocess_anomaly_corpus.py --raw_dir path/to/raw --output_dir data/processed/G2_v2
  python preprocess_anomaly_corpus.py --prasad other/Prasad_v2.csv   # override one source
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from url_features_enriched import extract_enriched_features, ENRICHED_FEATURE_NAMES

SEED = 42
LABEL_CONVENTION = "0 = phishing, 1 = legitimate"


# --------------------------------------------------------------------------
# Stage 1 — source loaders (each normalises to url, label, source)
# --------------------------------------------------------------------------
def load_prasad(path):
    df = pd.read_csv(path, usecols=["URL", "label"]).rename(columns={"URL": "url"})
    df["label"] = df["label"].astype(int)              # already 0=phish, 1=legit
    df["source"] = "phiusiil"
    return df[["url", "label", "source"]]


def load_jushnu(path):
    df = pd.read_csv(path)                              # columns: url, type
    df["label"] = (df["type"].str.strip().str.lower()
                   .map({"phishing": 0, "legitimate": 1}))
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["source"] = "jushnu"
    return df[["url", "label", "source"]]


def load_rachana(path):
    df = pd.read_csv(path).dropna(subset=["ClassLabel"]).rename(columns={"URL": "url"})
    # verified against the confirmed reference: matches Prasad as-is (NOT inverted)
    df["label"] = df["ClassLabel"].astype(int)
    df["source"] = "rachana"
    return df[["url", "label", "source"]]


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main(prasad_p, jushnu_p, rachana_p, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "seed": SEED, "label_convention": LABEL_CONVENTION}

    # --- Stage 1-2: combine, align, dedup ---
    parts = [load_prasad(prasad_p), load_jushnu(jushnu_p), load_rachana(rachana_p)]
    report["per_source"] = {
        p["source"].iloc[0]: {
            "rows": len(p),
            "phishing_0": int((p["label"] == 0).sum()),
            "legit_1": int((p["label"] == 1).sum()),
        } for p in parts}

    combined = pd.concat(parts, ignore_index=True)
    combined["url"] = combined["url"].astype(str).str.strip()
    n_raw = len(combined)

    # drop URLs appearing with conflicting labels across sources (ambiguous truth)
    nun = combined.groupby("url")["label"].transform("nunique")
    n_conflict = int(combined[nun > 1]["url"].nunique())
    combined = combined[nun == 1].drop_duplicates(subset=["url"]).reset_index(drop=True)

    report["rows_before_dedup"] = n_raw
    report["conflicting_urls_dropped"] = n_conflict
    report["rows_after_dedup"] = len(combined)

    # --- Stage 3: enriched feature extraction (shared extractor) ---
    feats = pd.DataFrame(
        [extract_enriched_features(u) for u in combined["url"]],
        columns=ENRICHED_FEATURE_NAMES,
    )
    feats["label"] = combined["label"].values

    # --- Stage 4: stratified 80/10/10 (seed 42) ---
    train, temp = train_test_split(
        feats, test_size=0.20, stratify=feats["label"], random_state=SEED)
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=SEED)

    # --- Stage 5: scale (fit on TRAIN features only) ---
    scaler = StandardScaler().fit(train[ENRICHED_FEATURE_NAMES].values)

    def scale(part):
        s = part.copy()
        s[ENRICHED_FEATURE_NAMES] = scaler.transform(part[ENRICHED_FEATURE_NAMES].values)
        return s

    train_s, val_s, test_s = scale(train), scale(val), scale(test)

    # --- Stage 6: legit-only subset (label==1 = LEGITIMATE) for novelty detection ---
    legit_only = train_s[train_s["label"] == 1][ENRICHED_FEATURE_NAMES]

    # --- Stage 7: save artefacts ---
    train_s.to_parquet(out / "url_features_train.parquet", index=False)
    val_s.to_parquet(out / "url_features_val.parquet", index=False)
    test_s.to_parquet(out / "url_features_test.parquet", index=False)
    legit_only.to_parquet(out / "url_features_legit_only.parquet", index=False)
    joblib.dump(scaler, out / "standard_scaler.pkl")

    with open(out / "feature_schema.json", "w") as f:
        json.dump({"features": ENRICHED_FEATURE_NAMES, "label": "label",
                   "n_features": len(ENRICHED_FEATURE_NAMES),
                   "convention": LABEL_CONVENTION}, f, indent=2)

    report.update({
        "n_features": len(ENRICHED_FEATURE_NAMES),
        "feature_names": ENRICHED_FEATURE_NAMES,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "train_label_dist": {int(k): int(v) for k, v in train["label"].value_counts().items()},
        "legit_only_train_rows": len(legit_only),
        "source_mix_final": combined["source"].value_counts().to_dict(),
        "note": "legit-only subset = label==1 (legitimate), corrected from prior inverted run",
    })
    with open(out / "g2_readiness_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- console summary ---
    print("=" * 60)
    print("G2 ANOMALY CORPUS — PREPROCESSING COMPLETE (v2, corrected labels)")
    print("=" * 60)
    for s, v in report["per_source"].items():
        print(f"  {s:<10}: {v['rows']:>8,} rows  ({v['phishing_0']:,} phish / {v['legit_1']:,} legit)")
    print("-" * 60)
    print(f"  Label convention     : {LABEL_CONVENTION}")
    print(f"  Rows before dedup    : {n_raw:,}")
    print(f"  Conflicting dropped  : {n_conflict:,}")
    print(f"  Rows after dedup     : {len(combined):,}")
    print(f"  Enriched features    : {len(ENRICHED_FEATURE_NAMES)}")
    print(f"  Split train/val/test : {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"  Train label dist     : {report['train_label_dist']}  (0=phish, 1=legit)")
    print(f"  Legit-only training  : {len(legit_only):,}  (label==1, genuinely legitimate)")
    print(f"  Source mix           : {report['source_mix_final']}")
    print(f"\n  Artefacts saved to   : {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="G2 anomaly corpus end-to-end preprocessing")
    ap.add_argument("--raw_dir", default="data/raw",
                    help="Directory holding the raw source CSVs (default: data/raw)")
    ap.add_argument("--prasad", default=None, help="Override path to PhiUSIIL/Prasad CSV (URL,label)")
    ap.add_argument("--jushnu", default=None, help="Override path to Jushnu CSV (url,type)")
    ap.add_argument("--rachana", default=None, help="Override path to Rachana CSV (URL,ClassLabel)")
    ap.add_argument("--output_dir", default="data/processed/G2_v2",
                    help="Where to write the prepared corpus (default: data/processed/G2_v2)")
    a = ap.parse_args()

    # Resolve each source: explicit override wins, else default filename under --raw_dir
    raw = Path(a.raw_dir)
    prasad_p  = a.prasad  or str(raw / "Prasad_v2.csv")
    jushnu_p  = a.jushnu  or str(raw / "Jushnu_K.csv")
    rachana_p = a.rachana or str(raw / "Rachana_P.csv")

    # Fail early with a clear message if a source file is missing
    for label, p in [("prasad", prasad_p), ("jushnu", jushnu_p), ("rachana", rachana_p)]:
        if not Path(p).exists():
            raise SystemExit(f"ERROR: {label} file not found at '{p}'. "
                             f"Check --raw_dir or pass --{label} explicitly.")

    main(prasad_p, jushnu_p, rachana_p, a.output_dir)