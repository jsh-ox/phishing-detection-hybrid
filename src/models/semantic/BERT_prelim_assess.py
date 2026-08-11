# %% [markdown]
# # BERT Fine-Tuning

# %% Setup and configuration
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --- Paths ---
DATA_DIR   = Path("data/processed/G1 Semantic")
TRAIN_PATH = DATA_DIR / "semantic_train.csv"
VAL_PATH   = DATA_DIR / "semantic_val.csv"
TEST_PATH  = DATA_DIR / "semantic_test.csv"
OUTPUT_DIR = Path("results/semantic/bert")

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

# %% [markdown]
# ## Assess — Casing retention (cased vs uncased BERT)

# %% Measure how much capitalisation remains in the corpus
def casing_stats(texts):
    texts = texts.dropna().astype(str)
    n = len(texts)
    has_upper = texts.str.contains(r"[A-Z]").mean()
    def upper_frac(s):
        letters = [c for c in s if c.isalpha()]
        return (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    upper_share = texts.map(upper_frac).mean()
    return n, has_upper, upper_share

n, has_upper, upper_share = casing_stats(train_df["text"])
print(f"Rows analysed              : {n:,}")
print(f"Rows containing ANY capital: {has_upper:.1%}")
print(f"Mean uppercase share       : {upper_share:.1%} of letters")
print()
if has_upper < 0.5:
    print(">> Most rows have NO capitals -> corpus is largely lowercased.")
    print(">> RECOMMENDATION: bert-base-uncased.")
else:
    print(">> Capitalisation is largely intact.")
    print(">> RECOMMENDATION: bert-base-cased.")

# %% [markdown]
# ## Assess 2 — Sequence length (`max_length`)
# Transformers truncate to `max_length`; anything beyond is discarded.

# %% Measure token-length distribution with the actual tokeniser
from transformers import AutoTokenizer

# temporary tokeniser just to measure lengths
_probe_model = "bert-base-uncased"
_tok = AutoTokenizer.from_pretrained(_probe_model)

# token counts on a sample — raise n for the full set
sample = train_df["text"].dropna().astype(str).sample(
    min(5000, len(train_df)), random_state=SEED)
lengths = sample.map(lambda t: len(_tok.encode(t, truncation=False)))

print("Token-length distribution (sample of {:,}):".format(len(sample)))
for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  {int(q*100)}th percentile: {int(lengths.quantile(q))} tokens")
print(f"  max: {int(lengths.max())} tokens")
print()
for cap in [128, 256, 384, 512]:
    covered = (lengths <= cap).mean()
    print(f"  max_length={cap:<3} would fully cover {covered:.1%} of emails")

# %% Visualise the token-length distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 4))
plt.hist(lengths.clip(upper=600), bins=60)
plt.axvline(256, color="r", ls="--", label="max_length=256")
plt.axvline(512, color="orange", ls="--", label="max_length=512 (BERT max)")
plt.xlabel("tokens per email"); plt.ylabel("count")
plt.title("Token-length distribution"); plt.legend(); plt.tight_layout(); plt.show()