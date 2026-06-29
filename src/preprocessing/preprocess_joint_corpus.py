"""
Pipeline C - Joint Evaluation Corpus (G3)
==========================================
Project: Hybrid ML for Phishing Detection

Produces the joint evaluation corpus from CEAS 2008, where each email yields
BOTH signals from the same object:
  - cleaned + redacted body text  -> semantic layer (matches G1 processing)
  - URL feature vectors            -> anomaly layer (matches G2 18-feature schema)

Outputs two linked tables (joined on email_id):
  joint_eval_<name>_text.parquet : email_id, text, label, n_urls
  joint_eval_<name>_urls.parquet : email_id, url_index, <18 scaled features>, is_null_url

Design decisions (documented for methodology):
  - URLs are extracted from the RAW body BEFORE redaction; the text field is then
    cleaned + redacted with the SAME functions used for the G1 semantic corpus.
  - URL features are computed with url_feature_extractor (a reimplementation of
    the PhiUSIIL definitions, ~95% parity - see dissertation limitation).
  - Features are scaled with the SAVED G2 StandardScaler (never refit) so email
    URLs sit in the same space the anomaly models learned.
  - Multi-URL emails: ALL URLs are kept as separate rows; aggregation (max
    anomaly score) happens at inference, per the project plan.
  - Null-URL emails: represented by a single row of zeros in SCALED space
    (= feature means, a neutral/non-anomalous point) tagged is_null_url=True.
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from url_feature_extractor import extract_url_features, FEATURE_ORDER

# reuse the G1 semantic cleaning + redaction functions
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pp", str(Path(__file__).parent / "preprocess_semantic_corpus.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

URL_RE = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')


def main(input_csv, scaler_path, output_dir, name="ceas"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(input_csv)
    report = {
        "source_file": Path(input_csv).name,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scaler_used": Path(scaler_path).name,
        "feature_parity_note": "URL features via reimplemented PhiUSIIL extractor (~95% parity)",
        "steps": {"loaded_emails": len(df)},
    }

    text_rows = []
    url_rows = []

    for email_id, row in df.iterrows():
        raw_body = str(row.get("body", "") or "")
        label = int(row["label"])

        # 1. extract URLs from RAW body (before redaction)
        urls = URL_RE.findall(raw_body)

        # 2. clean + redact body text (same as G1 semantic pipeline)
        text = pp.clean_text(raw_body)
        counters = {"<EMAIL>": 0, "<PHONE>": 0, "<SSN>": 0, "<CARD>": 0, "<URL>": 0}
        text = pp.redact_pii(text, counters)

        text_rows.append({
            "email_id": email_id,
            "text": text,
            "label": label,
            "n_urls": len(urls),
        })

        # 3. URL feature engineering
        if urls:
            for j, u in enumerate(urls):
                feats = extract_url_features(u)
                rec = {"email_id": email_id, "url_index": j, "is_null_url": False}
                rec.update({f: feats[f] for f in FEATURE_ORDER})
                url_rows.append(rec)
        else:
            # null-URL email: placeholder, filled with scaled-space zeros below
            url_rows.append({
                "email_id": email_id, "url_index": 0, "is_null_url": True,
                **{f: 0.0 for f in FEATURE_ORDER}
            })

    text_df = pd.DataFrame(text_rows)
    url_df = pd.DataFrame(url_rows)

    # 4. scale real URL features with the saved G2 scaler; null rows -> zeros (means)
    real_mask = ~url_df["is_null_url"].values
    feat_matrix = url_df.loc[real_mask, FEATURE_ORDER].values.astype(float)
    scaled = scaler.transform(feat_matrix)
    url_df.loc[real_mask, FEATURE_ORDER] = scaled
    # null-URL rows: scaled-space zeros (= feature means, neutral) - already 0.0

    # 5. save artefacts
    text_path = out / f"joint_eval_{name}_text.parquet"
    url_path = out / f"joint_eval_{name}_urls.parquet"
    text_df.to_parquet(text_path, index=False)
    url_df.to_parquet(url_path, index=False)

    # 6. validation + report
    assert len(text_df) == len(df), "text rows must equal email count"
    assert url_df["email_id"].nunique() == len(df), "every email must appear in URL table"
    report["steps"]["total_urls_extracted"] = int(real_mask.sum())
    report["steps"]["null_url_emails"] = int((~real_mask).sum())
    report["steps"]["class_distribution"] = {
        int(k): int(v) for k, v in text_df["label"].value_counts().items()}
    report["steps"]["url_presence_rate"] = round(
        (text_df["n_urls"] > 0).mean(), 4)
    report["steps"]["avg_urls_per_email"] = round(text_df["n_urls"].mean(), 2)
    with open(out / f"g3_{name}_readiness_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print(f"PIPELINE C COMPLETE - Joint Evaluation Corpus (G3) [{name}]")
    print("=" * 60)
    print(f"Emails processed     : {len(text_df):,}")
    print(f"Total URLs extracted : {int(real_mask.sum()):,}")
    print(f"Null-URL emails      : {int((~real_mask).sum()):,}")
    print(f"URL presence rate    : {report['steps']['url_presence_rate']:.1%}")
    print(f"Avg URLs / email     : {report['steps']['avg_urls_per_email']}")
    print(f"Class distribution   : {report['steps']['class_distribution']}")
    print()
    print(f"Text table : {text_path.name}  ({len(text_df):,} rows)")
    print(f"URL table  : {url_path.name}  ({len(url_df):,} rows)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("scaler_path")
    ap.add_argument("output_dir")
    ap.add_argument("--name", default="ceas")
    args = ap.parse_args()
    main(args.input_csv, args.scaler_path, args.output_dir, args.name)
