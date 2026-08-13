# # Fusion Step 1b — Anomaly scores on the G3 hyrbid data
# Runs each trained anomaly model over the CEAS URL features, then aggregates
# per email: an email's anomaly score = the MAX over its URLs (most suspicious
# link wins). Null-URL emails get the neutral score.

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import re
from sklearn.metrics import roc_auc_score, average_precision_score


G3_DIR = Path("data/processed/G3 Hybrid")
URL_PATH = G3_DIR / "joint_eval_ceas_urls.parquet"
TEXT_PATH = G3_DIR / "joint_eval_ceas_text_redacted.parquet"

ANOM_DIR = Path("results/anomaly")
OUT_DIR = Path("results/hybrid/g3_scores")
OUT_DIR.mkdir(parents=True, exist_ok=True)

url_df = pd.read_parquet(URL_PATH)
text_df = pd.read_parquet(TEXT_PATH)
# Force text_df email_id to int
text_df["email_id"] = text_df["email_id"].astype(int)
text_df["label"] = text_df["label"].astype(int)
print("text_df email_id dtype:", text_df["email_id"].dtype)

FEATURES = ['URLLength','DomainLength','IsDomainIP','TLDLength','NoOfSubDomain',
    'HasObfuscation','NoOfObfuscatedChar','ObfuscationRatio','NoOfLettersInURL',
    'LetterRatioInURL','NoOfDegitsInURL','DegitRatioInURL','NoOfEqualsInURL',
    'NoOfQMarkInURL','NoOfAmpersandInURL','NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL','IsHTTPS']

X_urls = url_df[FEATURES].values.astype("float32")
print("URL rows:", X_urls.shape, "| emails:", url_df['email_id'].nunique())


# ## Score each URL with the three anomaly models

# --- Isolation Forest ---
iso = joblib.load(ANOM_DIR / "isolation_forest" / "isolation_forest.pkl")
url_df["iso_score"] = -iso.score_samples(X_urls)

# --- Feedforward autoencoder (reconstruction MSE) ---
ff = keras.models.load_model(ANOM_DIR / "autoencoder_ff" / "autoencoder_ff.keras")
recon = ff.predict(X_urls, batch_size=512, verbose=0)
url_df["ff_score"] = np.mean((X_urls - recon) ** 2, axis=1)

print("Per-URL scores computed for Isolation Forest and feedforward AE.")
print(url_df[["iso_score","ff_score"]].describe().round(4))

# %% Aggregate per email: MAX over each email's URLs (most suspicious link)
real = url_df[~url_df["is_null_url"]]
null_ids = set(url_df[url_df["is_null_url"]]["email_id"]) - set(real["email_id"])

def aggregate(col):
    agg = real.groupby("email_id")[col].max()
    # emails with only null URLs -> neutral
    neutral = real[col].median()
    for eid in null_ids:
        agg.loc[eid] = neutral
    return agg.reset_index() 

anom = text_df[["email_id", "label"]].copy()
print("anom email_id dtype:", anom["email_id"].dtype)

iso_agg = aggregate("iso_score").rename(columns={"iso_score": "iso"})
ff_agg = aggregate("ff_score").rename(columns={"ff_score": "ff"})

anom = anom.merge(iso_agg, on="email_id", how="left")
anom = anom.merge(ff_agg,  on="email_id", how="left")

print("non-null iso:", anom["iso"].notna().sum(), "/", len(anom))
print("non-null ff :", anom["ff"].notna().sum(), "/", len(anom))
print(anom[["iso","ff"]].describe().round(4))

# --- LSTM autoencoder ---
# The LSTM AE was trained on raw URL character sequences, which are not in this
# feature table. Scoring it requires encoding URLs to char sequences

CEAS_RAW = Path("data/raw/CEAS_08.csv")
URL_RE = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')

# Re-extract URLs per email, in order, matching the original build.
ceas = pd.read_csv(CEAS_RAW)
rows = []
for email_id, row in ceas.iterrows():
    raw_body = str(row.get("body", "") or "")
    for j, u in enumerate(URL_RE.findall(raw_body)):
        rows.append({"email_id": email_id, "url_index": j, "url": u})
raw_urls = pd.DataFrame(rows)
print(f"Re-extracted {len(raw_urls):,} URLs across {raw_urls['email_id'].nunique():,} emails")

# URL count should match the real (non-null) rows in the feature table
n_real_features = (~url_df["is_null_url"]).sum()
print(f"Feature table real URLs: {n_real_features:,}  |  re-extracted: {len(raw_urls):,}")
if len(raw_urls) != n_real_features:
    print("mismatch — alignment may differ; check CEAS_RAW is the same file used in the G3 build")

# --- Encode to character sequences with the saved vocab ---
meta = json.load(open(ANOM_DIR / "autoencoder_lstm" / "lstm_data" / "char_vocab.json"))
vocab, MAX_LEN, PAD, OOV = meta["vocab"], meta["max_len"], meta["pad_idx"], meta["oov_idx"]

def encode(urls):
    out = np.full((len(urls), MAX_LEN), PAD, dtype=np.int32)
    for i, u in enumerate(urls):
        for j, c in enumerate(str(u)[:MAX_LEN]):
            out[i, j] = vocab.get(c, OOV)
    return out

seqs = encode(raw_urls["url"].tolist())

# --- Score with the LSTM autoencoder ---
lstm = keras.models.load_model(ANOM_DIR / "autoencoder_lstm" / "autoencoder_lstm.keras")
probs = lstm.predict(seqs, batch_size=256, verbose=0)
idx = np.take_along_axis(probs, seqs[..., None], axis=-1).squeeze(-1)
ce = -np.log(np.clip(idx, 1e-9, 1.0))
mask = (seqs != 0).astype("float32")
raw_urls["lstm_score"] = (ce * mask).sum(1) / np.maximum(mask.sum(1), 1)

# --- Aggregate per email (max), merge into the anomaly scores table ---
lstm_agg = raw_urls.groupby("email_id")["lstm_score"].max()
# ensure both keys are the same dtype before merging
anom["email_id"] = anom["email_id"].astype(int)
lstm_agg.index = lstm_agg.index.astype(int)
anom = anom.merge(lstm_agg.rename("lstm"), on="email_id", how="left")
anom["lstm"] = anom["lstm"].fillna(anom["lstm"].median())
print("LSTM AE scores added.")
print(anom[["iso","ff","lstm"]].describe().round(4))

# Save the complete anomaly scores
anom.to_parquet(OUT_DIR / "anomaly_scores_ceas.parquet", index=False)
print("Saved complete anomaly scores:", anom.shape)


# How do the anomaly models perform on G3 Hybrid data individually?
y = anom["label"].astype(int).values
print("Anomaly models on G3 CEAS (individual):")
print(f"{'model':<8}{'AUC-ROC':>10}{'AUC-PR':>9}")
for m in ["iso", "ff", "lstm"]:
    s = anom[m].values
    print(f"{m:<8}{roc_auc_score(y, s):>10.4f}{average_precision_score(y, s):>9.4f}")