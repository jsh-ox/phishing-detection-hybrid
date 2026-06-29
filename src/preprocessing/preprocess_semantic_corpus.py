"""
Pipeline A - Semantic Training Corpus Preprocessing
====================================================
Project: Hybrid ML for Phishing Detection (G1)

Transforms a raw labelled phishing/safe email CSV into a clean, redacted,
deduplicated, binary-labelled corpus ready for transformer fine-tuning.

Order of operations (matches documented data-readiness Pipeline A):
  1. Structural cleaning   - drop index col, nulls, exact duplicates
  2. Length sanity cap     - remove corrupted/trivial rows
  3. Text cleaning         - normalise whitespace / escaped newlines / encoding
  4. PII redaction         - regex pass: emails, phones, card-like, SSN-like, URLs
  5. Label encoding        - Safe=0, Phishing=1
  6. Class-balance report  - flag if imbalance exceeds 70:30
  7. Save artefact + report

NOTE: regex PII redaction is a FIRST PASS for structured identifiers only.
A Presidio NER pass is the planned SECOND PASS for free-text names (see report).
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MIN_LEN = 10          # drop rows shorter than this (chars)
MAX_LEN = 50_000      # drop rows longer than this (likely corrupted/concatenated)
IMBALANCE_THRESHOLD = 0.70   # flag SMOTE if majority class exceeds this fraction
RANDOM_SEED = 42

# ----------------------------------------------------------------------
# PII patterns (first-pass, structured identifiers)
# ----------------------------------------------------------------------
PII_PATTERNS = {
    "<EMAIL>": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "<PHONE>": re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}"),
    "<SSN>":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "<CARD>":  re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}
# URLs are redacted to <URL> for the semantic corpus body text.
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text: str) -> str:
    """Normalise whitespace, escaped newlines, and stray control characters."""
    if not isinstance(text, str):
        return ""
    # Decode common escaped sequences that appear literally in the CSV
    text = text.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ")
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def redact_pii(text: str, counters: dict) -> str:
    """Replace structured PII and URLs with tokens. Mutates counters in place."""
    # Order matters: SSN before CARD (SSN is a digit subsequence), then phone, email, url
    for token in ("<SSN>", "<CARD>", "<PHONE>", "<EMAIL>"):
        pattern = PII_PATTERNS[token]
        text, n = pattern.subn(token, text)
        counters[token] += n
    text, n = URL_PATTERN.subn("<URL>", text)
    counters["<URL>"] += n
    return text


def main(input_path: str, output_dir: str):
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "source_file": in_path.name,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "steps": {},
    }

    # --- Load ---
    df = pd.read_csv(in_path)
    report["steps"]["loaded_rows"] = len(df)

    # --- 1. Structural cleaning ---
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.rename(columns={"Email Text": "text", "Email Type": "label_raw"})

    n0 = len(df)
    df = df.dropna(subset=["text"])
    report["steps"]["removed_null_bodies"] = n0 - len(df)

    n1 = len(df)
    df = df.drop_duplicates(subset=["text"])
    report["steps"]["removed_exact_duplicates"] = n1 - len(df)

    # --- 2. Length sanity cap ---
    lengths = df["text"].str.len()
    n2 = len(df)
    df = df[(lengths >= MIN_LEN) & (lengths <= MAX_LEN)]
    report["steps"]["removed_length_outliers"] = n2 - len(df)
    report["steps"]["length_bounds"] = {"min_chars": MIN_LEN, "max_chars": MAX_LEN}

    # --- 3. Text cleaning ---
    df["text"] = df["text"].map(clean_text)
    # Cleaning can create empties or new duplicates; re-check
    n3 = len(df)
    df = df[df["text"].str.len() >= MIN_LEN]
    df = df.drop_duplicates(subset=["text"])
    report["steps"]["removed_post_clean_empty_or_dup"] = n3 - len(df)

    # --- 4. PII redaction (first pass) ---
    counters = {"<EMAIL>": 0, "<PHONE>": 0, "<SSN>": 0, "<CARD>": 0, "<URL>": 0}
    df["text"] = df["text"].map(lambda t: redact_pii(t, counters))
    report["steps"]["pii_redaction_counts"] = counters
    report["steps"]["pii_method"] = "regex first-pass (structured identifiers + URLs)"
    report["steps"]["pii_second_pass_planned"] = (
        "Microsoft Presidio NER for free-text names - not yet applied"
    )

    # --- 5. Label encoding ---
    label_map = {"Safe Email": 0, "Phishing Email": 1}
    df["label"] = df["label_raw"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df = df[["text", "label"]]

    # --- 6. Class-balance assessment ---
    counts = df["label"].value_counts().to_dict()
    safe_n = counts.get(0, 0)
    phish_n = counts.get(1, 0)
    total = safe_n + phish_n
    majority_frac = max(safe_n, phish_n) / total if total else 0
    report["steps"]["class_distribution"] = {
        "safe_0": int(safe_n),
        "phishing_1": int(phish_n),
        "majority_fraction": round(majority_frac, 4),
        "smote_recommended": bool(majority_frac > IMBALANCE_THRESHOLD),
    }
    report["steps"]["final_rows"] = len(df)

    # --- 7. Save artefact + report ---
    artefact_path = out_dir / "semantic_corpus_clean.csv"
    df.to_csv(artefact_path, index=False)

    report_path = out_dir / "g1_data_readiness_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Console summary ---
    print("=" * 60)
    print("PIPELINE A COMPLETE - Semantic Training Corpus (G1)")
    print("=" * 60)
    print(f"Source rows loaded         : {report['steps']['loaded_rows']:,}")
    print(f"Removed (null bodies)      : {report['steps']['removed_null_bodies']:,}")
    print(f"Removed (exact duplicates) : {report['steps']['removed_exact_duplicates']:,}")
    print(f"Removed (length outliers)  : {report['steps']['removed_length_outliers']:,}")
    print(f"Removed (post-clean)       : {report['steps']['removed_post_clean_empty_or_dup']:,}")
    print(f"Final clean rows           : {report['steps']['final_rows']:,}")
    print()
    print("PII redactions applied:")
    for k, v in counters.items():
        print(f"   {k:<8}: {v:,}")
    print()
    print("Class distribution:")
    print(f"   Safe (0)     : {safe_n:,}")
    print(f"   Phishing (1) : {phish_n:,}")
    print(f"   Majority frac: {majority_frac:.1%}")
    print(f"   SMOTE needed : {report['steps']['class_distribution']['smote_recommended']}")
    print()
    print(f"Artefact saved : {artefact_path}")
    print(f"Report saved   : {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to raw phishing email CSV")
    parser.add_argument("output_dir", help="Directory for cleaned artefact + report")
    args = parser.parse_args()
    main(args.input, args.output_dir)
