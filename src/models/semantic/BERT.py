# %% [markdown]
# - Model: bert-base-cased
# - max_length: 512 (78% data coverage, 512 is max for BERT)

# %% Setup and configuration
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, 
    EarlyStoppingCallback, AutoModelForSequenceClassification
)

from datasets import Dataset

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)

import matplotlib.pyplot as plt

# --- Paths ---
DATA_DIR   = Path("data/processed/G1 Semantic")
TRAIN_PATH = DATA_DIR / "semantic_train.csv"
VAL_PATH   = DATA_DIR / "semantic_val.csv"
TEST_PATH  = DATA_DIR / "semantic_test.csv"
OUTPUT_DIR = Path("results/semantic/bert")

# %% Training configuration
MODEL_NAME    = "bert-base-cased"
MAX_LENGTH    = 512
NUM_LABELS    = 2

# --- Confirm GPU ---
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# %% Load the splits and check
train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("Shapes:", train_df.shape, val_df.shape, test_df.shape)
print("Columns:", list(train_df.columns))
print("\nLabel balance (train):")
print(train_df["label"].value_counts(normalize=True).round(3).to_dict())
print("\nSample text:")
print(repr(train_df["text"].iloc[0])[:300])

# Memory-conscious settings for 8 GB VRAM
BATCH_SIZE           = 4
GRAD_ACCUM_STEPS     = 4
LEARNING_RATE        = 2e-5
EPOCHS               = 4
WEIGHT_DECAY         = 0.01
WARMUP_RATIO         = 0.1
FP16                 = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% Tokenise the splits
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def to_hf_dataset(df):
    return Dataset.from_pandas(
        df[["text", "label"]].dropna().reset_index(drop=True),
        preserve_index=False,
    )

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

train_ds = to_hf_dataset(train_df).map(tokenize_batch, batched=True, remove_columns=["text"])
val_ds   = to_hf_dataset(val_df).map(tokenize_batch,   batched=True, remove_columns=["text"])
test_ds  = to_hf_dataset(test_df).map(tokenize_batch,  batched=True, remove_columns=["text"])

print("Tokenised:", len(train_ds), len(val_ds), len(test_ds))

# %% Load the model with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)
print(f"Loaded {MODEL_NAME} — {sum(p.numel() for p in model.parameters()):,} parameters")

# %% Metrics for evaluation during training
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc_roc": roc_auc_score(labels, probs),
        "auc_pr": average_precision_score(labels, probs),
    }

# %% Training arguments
data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    fp16=FP16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,
    logging_steps=50,
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# %% Train
train_result = trainer.train()
print("\nTraining complete.")
print("Best model reloaded (load_best_model_at_end=True).")

# %% [markdown]
# ## Evaluation on the held-out test set
# Validation metrics guided model selection, so the TEST set is the unbiased
# figure to report. Same metric suite as the anomaly layer for a clean comparison.

# %% Predict on the test set
pred_output = trainer.predict(test_ds)
test_logits = pred_output.predictions
test_labels = pred_output.label_ids

# probability of the phishing class (index 1)
test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

print("Test predictions generated:", test_probs.shape[0], "rows")

# %% Compute the full metric suite
tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
metrics = {
    "accuracy":  round(accuracy_score(test_labels, test_preds), 4),
    "precision": round(precision_score(test_labels, test_preds, zero_division=0), 4),
    "recall":    round(recall_score(test_labels, test_preds, zero_division=0), 4),
    "f1":        round(f1_score(test_labels, test_preds, zero_division=0), 4),
    "auc_roc":   round(roc_auc_score(test_labels, test_probs), 4),
    "auc_pr":    round(average_precision_score(test_labels, test_probs), 4),
    "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}

print("=" * 50)
print("BERT — TEST SET METRICS")
print("=" * 50)
for k in ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr", "false_positive_rate"]:
    print(f"  {k:<20}: {metrics[k]}")
print(f"  confusion matrix    : {metrics['confusion_matrix']}")

# %% [markdown]
# ### Visualise: confusion matrix, ROC, precision-recall, and score distribution

fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# --- Confusion matrix ---
cm = np.array([[tn, fp], [fn, tp]])
im = ax[0, 0].imshow(cm, cmap="Blues")
ax[0, 0].set_xticks([0, 1]); ax[0, 0].set_xticklabels(["legit", "phish"])
ax[0, 0].set_yticks([0, 1]); ax[0, 0].set_yticklabels(["legit", "phish"])
ax[0, 0].set_xlabel("Predicted"); ax[0, 0].set_ylabel("Actual")
ax[0, 0].set_title("Confusion matrix")
for i in range(2):
    for j in range(2):
        ax[0, 0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)

# --- ROC curve ---
fpr, tpr, _ = roc_curve(test_labels, test_probs)
ax[0, 1].plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']}")
ax[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
ax[0, 1].set_xlabel("False positive rate"); ax[0, 1].set_ylabel("True positive rate")
ax[0, 1].set_title("ROC curve"); ax[0, 1].legend()

# --- Precision-Recall curve ---
prec, rec, _ = precision_recall_curve(test_labels, test_probs)
ax[1, 0].plot(rec, prec, label=f"AUC-PR = {metrics['auc_pr']}")
ax[1, 0].set_xlabel("Recall"); ax[1, 0].set_ylabel("Precision")
ax[1, 0].set_title("Precision-Recall curve"); ax[1, 0].legend()

# --- Predicted-probability distribution by class ---
ax[1, 1].hist(test_probs[test_labels == 0], bins=50, alpha=0.6, label="legitimate", density=True)
ax[1, 1].hist(test_probs[test_labels == 1], bins=50, alpha=0.6, label="phishing", density=True)
ax[1, 1].axvline(0.5, color="k", ls="--", lw=1, label="threshold 0.5")
ax[1, 1].set_xlabel("P(phishing)"); ax[1, 1].set_ylabel("density")
ax[1, 1].set_title("Predicted probability by class"); ax[1, 1].legend()

plt.tight_layout(); plt.show()

# %% [markdown]
# ### Error analysis

# %% Surface misclassified examples
test_texts = test_df["text"].dropna().reset_index(drop=True)
fp_idx = np.where((test_preds == 1) & (test_labels == 0))[0]  # legit flagged as phish
fn_idx = np.where((test_preds == 0) & (test_labels == 1))[0]  # phish missed

print(f"False positives (legit -> phish): {len(fp_idx)}")
print(f"False negatives (phish -> legit): {len(fn_idx)}")
print()
print("--- Sample FALSE POSITIVES (most confident mistakes) ---")
for i in fp_idx[np.argsort(-test_probs[fp_idx])][:3]:
    print(f"[P(phish)={test_probs[i]:.3f}] {repr(test_texts.iloc[i])[:200]}")
print()
print("--- Sample FALSE NEGATIVES (most confident mistakes) ---")
for i in fn_idx[np.argsort(test_probs[fn_idx])][:3]:
    print(f"[P(phish)={test_probs[i]:.3f}] {repr(test_texts.iloc[i])[:200]}")

# %% Save artefacts (model, per-row scores, report)

# fine-tuned model + tokeniser
trainer.save_model(str(OUTPUT_DIR / "model"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "model"))

# per-row test scores — needed by the fusion layer later
pd.DataFrame({
    "semantic_score": test_probs,
    "label": test_labels,
}).to_parquet(OUTPUT_DIR / "bert_scores_test.parquet", index=False)

# evaluation report (matches anomaly-layer format)
report = {
    "model": "bert-base-cased",
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "config": {"max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
               "grad_accum": GRAD_ACCUM_STEPS, "effective_batch": BATCH_SIZE * GRAD_ACCUM_STEPS,
               "learning_rate": LEARNING_RATE, "epochs": EPOCHS, "fp16": FP16},
    "test_metrics": metrics,
}
with open(OUTPUT_DIR / "bert_eval_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Saved: model/, bert_scores_test.parquet, bert_eval_report.json")
print(f"  -> {OUTPUT_DIR.resolve()}")