"""
Anomaly Layer — G3 Evaluation v2
============================================================================
Project: Hybrid ML for Phishing Detection

Scores the three anomaly models on the G3 joint-evaluation set
(out-of-distribution generalisation test).

Pipeline:
  1. Extract URLs from email bodies (same regex as the joint-corpus build)
  2. Enriched features via the extractor helper
  3. Scale with the saved G2 v2 scaler
  4. Score with Isolation Forest + feedforward AE (feature-based),
     and the LSTM AE (character-sequence-based)
  5. Aggregate per email: MAX over the email's URLs; null-URL emails -> neutral
  6. Evaluate each model (AUC-ROC / AUC-PR / F1 etc.)

Label convention: 0=phishing, 1=legitimate. Detection target = phishing.

Outputs (into OUTPUT_DIR):
  anomaly_scores_g3.parquet   email_id, label, iso, ff, lstm  (per-email)
  anomaly_g3_report.json        per-model metrics
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
import sys
from tensorflow import keras
from sklearn.metrics import roc_auc_score, average_precision_score

import joblib
import numpy as np
import pandas as pd

from src.preprocessing.url_features_enriched import extract_enriched_features, ENRICHED_FEATURE_NAMES
from src.models.anomaly.anomaly_eval import to_phishing_target, evaluate_all_strategies

# --- CONFIG ---
INPUT_CSV = Path("data/raw/Phishing_Email.csv")
TEXT_COL  = "Email Text"
LABEL_COL_NAME = "Email Type"
LABEL_MAP = {"Phishing Email": 0, "Safe Email": 1}
CEAS_CSV   = Path("data/raw/CEAS_08.csv")
SA_CSV   = Path("data/raw/SpamAssassin.csv")
G2_DIR     = Path("data/processed/G2_v2")
MODEL_DIR  = Path("results/anomaly")
LSTM_META  = Path("data/processed/G2_v2/lstm_data/char_vocab.json")
OUTPUT_DIR = Path("results/hybrid/anomaly_g3_scores")

# same regex as preprocess_joint_corpus.py (URL/feature alignment)
URL_RE = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
TARGET_FPR = 0.10
# ======================================================


def extract_urls_generic(csv_path, text_col, label_col, label_map):
    df = pd.read_csv(csv_path)
    rows, email_meta = [], []
    for email_id, row in df.iterrows():
        text = str(row.get(text_col, "") or "")
        raw = row.get(label_col)
        label = label_map.get(str(raw).strip(), -1)   # -> 0=phish, 1=legit
        urls = URL_RE.findall(text)
        email_meta.append({"email_id": email_id, "label": label, "n_urls": len(urls)})
        for j, u in enumerate(urls):
            rows.append({"email_id": email_id, "url_index": j, "url": u})
    return pd.DataFrame(rows), pd.DataFrame(email_meta)


def aggregate_per_email(url_df, email_meta, score_col):
    """MAX over each email's URLs; emails with no URL -> neutral (median score)."""
    agg = url_df.groupby("email_id")[score_col].max()
    neutral = float(url_df[score_col].median())
    out = email_meta[["email_id"]].copy()
    out[score_col] = out["email_id"].map(agg).fillna(neutral)
    return out.set_index("email_id")[score_col]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. extract URLs ---
    url_df, email_meta = extract_urls_generic(INPUT_CSV, TEXT_COL, LABEL_COL_NAME, LABEL_MAP)
    email_meta["label"] = email_meta["label"].astype(int)
    print(f"G3 DATA: {len(email_meta):,} emails | {len(url_df):,} URLs "
          f"({(email_meta['n_urls'] == 0).sum():,} emails have no URL)")

    # --- 2. enriched features via helper script) ---
    feats = pd.DataFrame(
        [extract_enriched_features(u) for u in url_df["url"]],
        columns=ENRICHED_FEATURE_NAMES,
    )

    # --- 3. scale with the saved G2 v2 scaler ---
    scaler = joblib.load(G2_DIR / "standard_scaler.pkl")
    X = scaler.transform(feats.values.astype(float))

    # --- 4a. feature-based models: Isolation Forest + feedforward AE ---
    iso = joblib.load(MODEL_DIR / "isolation_forest" / "isolation_forest.pkl")
    url_df["iso"] = -iso.score_samples(X)


    ff = keras.models.load_model(MODEL_DIR / "autoencoder_ff" / "autoencoder_ff.keras")
    recon = ff.predict(X, batch_size=512, verbose=0)
    url_df["ff"] = np.mean(np.square(X - recon), axis=1)

    # --- 4b. LSTM AE: character sequences ---
    meta = json.load(open(LSTM_META))
    vocab, MAX_LEN, PAD, OOV = meta["vocab"], meta["max_len"], meta["pad"], meta["oov"]

    def encode(urls):
        out = np.full((len(urls), MAX_LEN), PAD, dtype=np.int32)
        for i, u in enumerate(urls):
            for j, c in enumerate(str(u)[:MAX_LEN]):
                out[i, j] = vocab.get(c, OOV)
        return out

    lstm = keras.models.load_model(MODEL_DIR / "autoencoder_lstm" / "autoencoder_lstm.keras")
    seqs = encode(url_df["url"].tolist())
    probs = lstm.predict(seqs, batch_size=256, verbose=0)
    idx = np.take_along_axis(probs, seqs[..., None], axis=-1).squeeze(-1)
    ce = -np.log(np.clip(idx, 1e-9, 1.0))
    mask = (seqs != 0).astype("float32")
    url_df["lstm"] = (ce * mask).sum(1) / np.maximum(mask.sum(1), 1)

    # --- 5. aggregate per email (max), assemble ---
    anom = email_meta[["email_id", "label"]].copy().set_index("email_id")
    for col in ["iso", "ff", "lstm"]:
        anom[col] = aggregate_per_email(url_df, email_meta, col)
    anom = anom.reset_index()
    anom.to_parquet(OUTPUT_DIR / "anomaly_scores_ceas.parquet", index=False)

    # --- 6. evaluate each model on data ---
    y_phish = to_phishing_target(anom["label"].values)
    # for a single-set eval we use the same scores for threshold + metrics;
    # AUC is threshold-independent (the headline); F1 uses the youden threshold.
    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": "G3 Hybrid",
        "label_convention": "0=phishing, 1=legitimate; detection target = phishing",
        "n_emails": len(anom),
        "models": {},
    }
    print("\n" + "=" * 56)
    print("ANOMALY MODELS ON G3")
    print("=" * 56)
    print(f"  {'model':<8}{'AUC-ROC':>10}{'AUC-PR':>9}")
    for col in ["iso", "ff", "lstm"]:
        s = anom[col].values
        auc = roc_auc_score(y_phish, s)
        ap = average_precision_score(y_phish, s)
        report["models"][col] = {"auc_roc": round(auc, 4), "auc_pr": round(ap, 4)}
        flag = "  <-- INVERTED (below 0.5)" if auc < 0.5 else ""
        print(f"  {col:<8}{auc:>10.4f}{ap:>9.4f}{flag}")
    print("=" * 56)

    with open(OUTPUT_DIR / "anomaly_g3_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {OUTPUT_DIR/'anomaly_scores_g3.parquet'}")
    print(f"       {OUTPUT_DIR/'anomaly_g3_report.json'}")

    # ============================================================
    # DIAGNOSTIC INSPECTION — verify G3 data prep before trusting metrics
    # ============================================================
    print("\n" + "="*60)
    print("G3 DATA PREP DIAGNOSTICS")
    print("="*60)

    # --- 1. Shape & coverage: did every email get scored? ---
    print("\n[1] Coverage")
    print(f"  emails in email_meta : {len(email_meta):,}")
    print(f"  emails in anom       : {len(anom):,}")
    print(f"  URLs extracted       : {len(url_df):,}")
    for col in ["iso", "ff", "lstm"]:
        n_nan = anom[col].isna().sum()
        print(f"  {col:<5} non-null: {anom[col].notna().sum():,} / {len(anom):,}"
              f"  ({'OK' if n_nan==0 else f'{n_nan} NaN — PROBLEM'})")

    # --- 2. Label sanity: dtype, values, balance ---
    print("\n[2] Labels")
    print(f"  label dtype       : {anom['label'].dtype}  ({'OK' if anom['label'].dtype!='object' else 'STRING — will break metrics'})")
    print(f"  label values      : {sorted(anom['label'].unique())}  (expect [0, 1])")
    print(f"  label balance     : {anom['label'].value_counts().to_dict()}  (0=phish, 1=legit)")
    phish_frac = (anom['label']==0).mean()
    print(f"  phishing fraction : {phish_frac:.3f}  ({'plausible' if 0.2<phish_frac<0.8 else 'SUSPICIOUS'})")

    # --- 3. Null-URL handling: how many emails had no URL, what did they get? ---
    print("\n[3] Null-URL emails")
    no_url = email_meta[email_meta['n_urls']==0]['email_id']
    print(f"  emails with no URL : {len(no_url):,} ({len(no_url)/len(anom):.1%})")
    if len(no_url):
        neutral_check = anom[anom['email_id'].isin(no_url)]
        for col in ["iso","ff","lstm"]:
            uniq = neutral_check[col].nunique()
            print(f"    {col}: {uniq} distinct value(s) among null-URL emails "
                  f"({'OK — all neutral' if uniq==1 else 'varies — check aggregation'})")

    # --- 4. Score distributions: are they sane, or degenerate/constant? ---
    print("\n[4] Score distributions (per model)")
    for col in ["iso", "ff", "lstm"]:
        s = anom[col]
        print(f"  {col:<5} min={s.min():.4f} med={s.median():.4f} max={s.max():.4f} "
              f"std={s.std():.4f}  {'<-- near-constant!' if s.std()<1e-6 else ''}")

    # --- 5. Directional sanity: do phishing emails score HIGHER (more anomalous)? ---
    print("\n[5] Directional check (phishing SHOULD score higher = more anomalous)")
    for col in ["iso", "ff", "lstm"]:
        phish_mean = anom[anom['label']==0][col].mean()   # label 0 = phishing
        legit_mean = anom[anom['label']==1][col].mean()   # label 1 = legit
        direction = "correct" if phish_mean > legit_mean else "INVERTED"
        print(f"  {col:<5} phish={phish_mean:.4f}  legit={legit_mean:.4f}  "
              f"diff={phish_mean-legit_mean:+.4f}  [{direction}]")

    # --- 6. Feature-level sanity on the raw extracted features (pre-scaling) ---
    print("\n[6] Feature sanity (CEAS enriched features, pre-scaling)")
    print(f"  feature matrix shape : {feats.shape}  (expect (?, 24))")
    print(f"  any NaN in features  : {feats.isna().any().any()}")
    print(f"  any inf in features  : {np.isinf(feats.values).any()}")
    # compare a couple of feature means to what training-era values looked like
    for fcol in ['has_https', 'url_entropy', 'suspicious_keyword_count']:
        if fcol in feats.columns:
            print(f"    {fcol:<26} CEAS mean = {feats[fcol].mean():.4f}")

    print("="*60 + "\n")

    # ============================================================
    # DATA SAMPLES — eyeball real rows to catch what stats hide
    # ============================================================
    pd.set_option ("display.max_colwidth", 60)  # keep URLs readable
    print("\n" + "="*60)
    print("REAL DATA SAMPLES")
    print("="*60)

    # --- A. Raw extracted URLs with their enriched features (a few columns) ---
    print("\n[A] Sample extracted URLs + selected features (pre-scaling)")
    show_feats = [c for c in ['has_https','url_entropy','suspicious_keyword_count',
                              'is_ip_host','subdomain_depth'] if c in feats.columns]
    sample_idx = url_df.sample(min(8, len(url_df)), random_state=1).index
    preview = url_df.loc[sample_idx, ['email_id','url']].copy()
    for f in show_feats:
        preview[f] = feats.loc[sample_idx, f].round(3).values
    print(preview.to_string(index=False))

    # --- B. Highest-scoring URLs per model (should look phishing-like) ---
    print("\n[B] Top-5 MOST ANOMALOUS URLs per model (should look suspicious)")
    for col in ["iso", "ff", "lstm"]:
        print(f"\n  -- {col} --")
        top = url_df.nlargest(5, col)[['url', col]]
        for _, r in top.iterrows():
            print(f"    [{r[col]:.3f}]  {str(r['url'])[:70]}")

    # --- C. Lowest-scoring URLs per model (should look legitimate) ---
    print("\n[C] Bottom-5 LEAST ANOMALOUS URLs per model (should look normal)")
    for col in ["iso", "ff", "lstm"]:
        print(f"\n  -- {col} --")
        bot = url_df.nsmallest(5, col)[['url', col]]
        for _, r in bot.iterrows():
            print(f"    [{r[col]:.3f}]  {str(r['url'])[:70]}")

    # --- D. Per-email aggregated view: a few phishing and a few legit emails ---
    print("\n[D] Sample emails: aggregated scores vs their true label")
    phish_sample = anom[anom['label']==0].sample(min(4,(anom['label']==0).sum()), random_state=2)
    legit_sample = anom[anom['label']==1].sample(min(4,(anom['label']==1).sum()), random_state=2)
    print("  PHISHING emails (label 0 — scores SHOULD be high):")
    print(phish_sample[['email_id','label','iso','ff','lstm']].round(3).to_string(index=False))
    print("  LEGITIMATE emails (label 1 — scores SHOULD be low):")
    print(legit_sample[['email_id','label','iso','ff','lstm']].round(3).to_string(index=False))

    print("="*60 + "\n")

    for col in ["iso","ff","lstm"]:
        s = anom[col].values
        y = (anom['label']==0).astype(int)  # 1=phishing
        auc = roc_auc_score(y, s)
        auc_flipped = roc_auc_score(y, -s)
        print(f"{col}: AUC={auc:.3f}  flipped={auc_flipped:.3f}")

    scaler = joblib.load("data/processed/G2_v2/standard_scaler.pkl")
    X_ceas_scaled = scaler.transform(feats.values.astype(float))  # scale CEAS
    g2 = pd.read_parquet("data/processed/G2_v2/url_features_train.parquet")
    g2_scaled = g2.drop(columns=['label']).values  # already scaled

    print(f"{'feature':<28}{'G2_mean':>9}{'CEAS_mean':>10}{'shift':>8}")
    for i, c in enumerate(ENRICHED_FEATURE_NAMES):
        g2m = g2_scaled[:, i].mean()          # ~0 (scaled)
        cm  = X_ceas_scaled[:, i].mean()      # CEAS in the SAME scaled space
        print(f"{c:<28}{g2m:>9.2f}{cm:>10.2f}{cm-g2m:>8.2f}")

    # Test BOTH label directions against the anomaly scores
    y_as_flipped = (anom['label']==0).astype(int)      # current: 1=phishing after your flip
    for col in ["iso","ff","lstm"]:
        s = anom[col].values
        auc_current = roc_auc_score(y_as_flipped, s)
        auc_other   = roc_auc_score(1 - y_as_flipped, s)
        print(f"{col}: current label dir AUC={auc_current:.3f} | opposite={auc_other:.3f}")

    # score the G2 TEST set through the SAME functions the G3 script uses
    g2_test = pd.read_parquet("data/processed/G2_v2/url_features_test.parquet")
    Xg2 = g2_test.drop(columns=['label']).values
    yg2 = (g2_test['label']==0).astype(int)   # 1=phishing
    iso_scores_g2 = -iso.score_samples(Xg2)
    print("IF on G2 test via G3 code path:", roc_auc_score(yg2, iso_scores_g2))

    def score_with_neutralised(suspects):
        """Re-score CEAS with `suspects` features set to 0 (= training mean in scaled space)."""
        Xa = X.copy()
        for f in suspects:
            Xa[:, ENRICHED_FEATURE_NAMES.index(f)] = 0.0

        # re-score both feature-based models on the ablated matrix
        iso_s = -iso.score_samples(Xa)
        recon = ff.predict(Xa, batch_size=512, verbose=0)
        ff_s = np.mean((Xa - recon) ** 2, axis=1)

        # aggregate per email (max), same as the eval
        tmp = url_df[['email_id']].copy()
        tmp['iso'], tmp['ff'] = iso_s, ff_s
        y = (email_meta.set_index('email_id')['label'] == 0).astype(int)  # 1=phishing
        out = {}
        for col in ['iso', 'ff']:
            agg = tmp.groupby('email_id')[col].max()
            agg = agg.reindex(email_meta['email_id']).fillna(tmp[col].median())
            out[col] = roc_auc_score(y.values, agg.values)
        return out

    # baseline (nothing removed) vs progressive ablation
    print("baseline (all features):      ", score_with_neutralised([]))
    print("without has_https:            ", score_with_neutralised(['has_https']))
    print("without has_https+domain_ent: ", score_with_neutralised(['has_https', 'domain_entropy']))
    print("without top-4 shifted:        ", score_with_neutralised(
        ['has_https', 'domain_entropy', 'num_dots', 'path_depth']))


if __name__ == "__main__":
    main()
