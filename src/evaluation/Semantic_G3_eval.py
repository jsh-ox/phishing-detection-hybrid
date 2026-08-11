# %% [markdown]
# # Hybrid fusion part 1 - Semantic scores on the G3 corpus
# Runs each over G3 hybrid test set to produce per-email P(phishing).

# %% Setup
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

import joblib

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix

G3_DIR    = Path("data/processed/G3 Hybrid")
TEXT_PATH = G3_DIR / "joint_eval_ceas_text_redacted.parquet"

SEM_DIR   = Path("results/semantic")
OUT_DIR   = Path("results/hybrid/g3_scores")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

text_df = pd.read_parquet(TEXT_PATH)
print("Dataset emails:", len(text_df), "| label dist:", text_df["label"].value_counts().to_dict())

# %% Score with the fine-tuned classifier models (BERT, RoBERTa)
def score_classifier(model_dir, texts, max_length=512, batch_size=16):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size].tolist()
            enc = tok(batch, truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p)
            if (i // batch_size) % 50 == 0:
                print(f"  {i}/{len(texts)}", end="\r")
    del model; torch.cuda.empty_cache()
    return np.array(probs)

ceas_texts = text_df["text"].fillna("").astype(str)

print("Scoring BERT...")
bert_scores = score_classifier(SEM_DIR / "bert" / "model", ceas_texts)
print("Scoring RoBERTa...")
roberta_scores = score_classifier(SEM_DIR / "roberta" / "model", ceas_texts)

# %% Score with SBERT + logistic regression
def score_sbert(sbert_name, clf_path, texts, batch_size=32):
    encoder = SentenceTransformer(sbert_name, device=DEVICE)
    emb = encoder.encode(texts.tolist(), batch_size=batch_size,
                         convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=True)
    clf = joblib.load(clf_path)
    del encoder; torch.cuda.empty_cache()
    return clf.predict_proba(emb)[:, 1]

print("Scoring SBERT...")
sbert_scores = score_sbert("sentence-transformers/all-mpnet-base-v2",
                           SEM_DIR / "sbert" / "sbert_logreg.pkl", ceas_texts)

# %% Save semantic scores aligned to email_id
semantic_scores = pd.DataFrame({
    "email_id": text_df["email_id"].values,
    "label": text_df["label"].values,
    "bert": bert_scores,
    "roberta": roberta_scores,
    "sbert": sbert_scores,
})
semantic_scores.to_parquet(OUT_DIR / "semantic_scores_ceas.parquet", index=False)
print("Saved semantic scores:", semantic_scores.shape)
print(semantic_scores[["bert","roberta","sbert"]].describe().round(3))

y = semantic_scores["label"].astype(int).values
print("Semantic models on G3 CEAS (out-of-distribution):")
print(f"{'model':<10}{'F1':>8}{'AUC-ROC':>10}{'AUC-PR':>9}{'FPR':>8}")
for m in ["bert", "roberta", "sbert"]:
    p = semantic_scores[m].values
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    f1 = f1_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    fpr = fp / (fp + tn) if (fp + tn) else 0
    print(f"{m:<10}{f1:>8.4f}{auc:>10.4f}{ap:>9.4f}{fpr:>8.4f}")