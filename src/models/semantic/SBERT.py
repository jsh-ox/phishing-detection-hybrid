# %% [markdown]
# # Sentence-BERT
# FROZEN pre-trained SBERT
# encodes email into a fixed-size embedding
# classifier predicts phishing/legitimate.

# %% Setup and configuration
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import joblib

from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression

DATA_DIR   = Path("data/processed/G1 Semantic")
TRAIN_PATH = DATA_DIR / "semantic_train.csv"
VAL_PATH   = DATA_DIR / "semantic_val.csv"
TEST_PATH  = DATA_DIR / "semantic_test.csv"
OUTPUT_DIR = Path("results/semantic/sbert")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# SBERT model
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# %% Load the splits
train_df = pd.read_csv(TRAIN_PATH).dropna(subset=["text"]).reset_index(drop=True)
val_df   = pd.read_csv(VAL_PATH).dropna(subset=["text"]).reset_index(drop=True)
test_df  = pd.read_csv(TEST_PATH).dropna(subset=["text"]).reset_index(drop=True)
print("Shapes:", train_df.shape, val_df.shape, test_df.shape)

# %% Encode emails into SBERT embeddings
encoder = SentenceTransformer(SBERT_MODEL, device=DEVICE)
print("Embedding dimension:", encoder.get_sentence_embedding_dimension())

def embed(texts):
    return encoder.encode(
        texts.tolist(),
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

X_train = embed(train_df["text"]); y_train = train_df["label"].values
X_val   = embed(val_df["text"]);   y_val   = val_df["label"].values
X_test  = embed(test_df["text"]);  y_test  = test_df["label"].values
print("Embedding matrices:", X_train.shape, X_val.shape, X_test.shape)

# %% [markdown]
# %% Save embeddings
np.savez_compressed(OUTPUT_DIR / "sbert_embeddings.npz",
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    X_test=X_test, y_test=y_test)
print("Embeddings cached to", OUTPUT_DIR / "sbert_embeddings.npz")

# %% Train the classifier on the embeddings
clf = LogisticRegression(
    max_iter=2000,
    C=1.0,
    class_weight="balanced",
    random_state=SEED,
)
clf.fit(X_train, y_train)

val_probs  = clf.predict_proba(X_val)[:, 1]
test_probs = clf.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)
print("Classifier trained.")

# %% Evaluate on the test set
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
metrics = {
    "accuracy":  round(accuracy_score(y_test, test_preds), 4),
    "precision": round(precision_score(y_test, test_preds, zero_division=0), 4),
    "recall":    round(recall_score(y_test, test_preds, zero_division=0), 4),
    "f1":        round(f1_score(y_test, test_preds, zero_division=0), 4),
    "auc_roc":   round(roc_auc_score(y_test, test_probs), 4),
    "auc_pr":    round(average_precision_score(y_test, test_probs), 4),
    "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}
print("=" * 50)
print("SBERT + LogReg — TEST SET METRICS")
print("=" * 50)
for k in ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "false_positive_rate"]:
    print(f"  {k:<20}: {metrics[k]}")
print(f"  confusion matrix    : {metrics['confusion_matrix']}")

# %% Visualise


fig, ax = plt.subplots(2, 2, figsize=(12, 10))
cm = np.array([[tn, fp], [fn, tp]])
ax[0,0].imshow(cm, cmap="Blues")
ax[0,0].set_xticks([0,1]); ax[0,0].set_xticklabels(["legit","phish"])
ax[0,0].set_yticks([0,1]); ax[0,0].set_yticklabels(["legit","phish"])
ax[0,0].set_xlabel("Predicted"); ax[0,0].set_ylabel("Actual"); ax[0,0].set_title("Confusion matrix")
for i in range(2):
    for j in range(2):
        ax[0,0].text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=13)
fpr, tpr, _ = roc_curve(y_test, test_probs)
ax[0,1].plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']}"); ax[0,1].plot([0,1],[0,1],"k--",lw=1)
ax[0,1].set_title("ROC curve"); ax[0,1].set_xlabel("FPR"); ax[0,1].set_ylabel("TPR"); ax[0,1].legend()
prec, rec, _ = precision_recall_curve(y_test, test_probs)
ax[1,0].plot(rec, prec, label=f"AUC-PR = {metrics['auc_pr']}")
ax[1,0].set_title("Precision-Recall"); ax[1,0].set_xlabel("Recall"); ax[1,0].set_ylabel("Precision"); ax[1,0].legend()
ax[1,1].hist(test_probs[y_test==0], bins=50, alpha=0.6, label="legitimate", density=True)
ax[1,1].hist(test_probs[y_test==1], bins=50, alpha=0.6, label="phishing", density=True)
ax[1,1].axvline(0.5, color="k", ls="--", lw=1, label="threshold 0.5")
ax[1,1].set_title("Predicted probability by class"); ax[1,1].set_xlabel("P(phishing)"); ax[1,1].legend()
plt.tight_layout(); plt.show()

# %% Save artefacts (classifier, per-row scores, report)

joblib.dump(clf, OUTPUT_DIR / "sbert_logreg.pkl")

pd.DataFrame({"semantic_score": test_probs, "label": y_test}).to_parquet(
    OUTPUT_DIR / "sbert_scores_test.parquet", index=False)

report = {
    "model": f"SBERT ({SBERT_MODEL}) + LogisticRegression",
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "config": {"sbert_model": SBERT_MODEL,
               "embedding_dim": int(X_train.shape[1]),
               "classifier": "LogisticRegression(C=1.0, class_weight=balanced)",
               "embeddings_normalized": True},
    "test_metrics": metrics,
}
with open(OUTPUT_DIR / "sbert_eval_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("Saved: sbert_logreg.pkl, sbert_scores_test.parquet, sbert_eval_report.json")