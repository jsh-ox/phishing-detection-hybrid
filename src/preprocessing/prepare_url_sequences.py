"""
LSTM Autoencoder — Character-Sequence Data Prep (G2 v2)
Encodes raw URLs from the combined corpus into character-index sequences
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
MAX_LEN = 128
PAD, OOV = 0, 1
CORPUS = Path("data") / "processed" / "G2_v2" / "url_corpus_combined.parquet"
OUT = Path("data/processed/G2_v2/lstm_data"); OUT.mkdir(exist_ok=True)

df = pd.read_parquet(CORPUS)
train_df, temp = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=SEED)
val_df, test_df = train_test_split(temp, test_size=0.50, stratify=temp['label'], random_state=SEED)

def build_vocab(urls):
    chars = sorted({c for u in urls for c in str(u)})
    return {c: i+2 for i, c in enumerate(chars)}

def encode(urls, vocab):
    out = np.full((len(urls), MAX_LEN), PAD, dtype=np.int32)
    for i, u in enumerate(urls):
        for j, c in enumerate(str(u)[:MAX_LEN]):
            out[i, j] = vocab.get(c, OOV)
    return out

vocab = build_vocab(train_df['url'].tolist())
train_legit = train_df[train_df['label'] == 1]

np.save(OUT/'seq_train_legit.npy', encode(train_legit['url'].tolist(), vocab))
np.save(OUT/'seq_val.npy',  encode(val_df['url'].tolist(), vocab))
np.save(OUT/'seq_test.npy', encode(test_df['url'].tolist(), vocab))
np.save(OUT/'val_labels.npy',  val_df['label'].values.astype(np.int32))
np.save(OUT/'test_labels.npy', test_df['label'].values.astype(np.int32))
json.dump({'vocab':vocab,'max_len':MAX_LEN,'pad':PAD,'oov':OOV,'vocab_size':len(vocab)+2,
           'convention':'0=phishing,1=legitimate'}, open(OUT/'char_vocab.json','w'), indent=2)

print("LSTM sequence prep complete")
print(f"  train/val/test : {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
print(f"  legit-only train: {len(train_legit):,}")
print(f"  vocab size     : {len(vocab)+2} | max_len: {MAX_LEN}")
