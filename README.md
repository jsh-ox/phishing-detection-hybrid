# Hybrid Machine Learning for Phishing Detection in Emails

A BSc artefact project investigating the effectiveness of **hybrid machine
learning techniques** — combining semantic language-model analysis with anomaly
detection — for identifying and flagging suspicious phishing content in emails.

> **Research question:** *To what extent are hybrid machine learning techniques —
> combining semantic language model analysis with anomaly detection — effective in
> identifying and flagging suspicious phishing content in emails?*

---

## Overview

The system is a four-stage hybrid pipeline:

1. **Preprocessing** — shared cleaning, redaction, and feature extraction.
2. **Semantic layer** — BERT, RoBERTa, and Sentence-BERT analyse email body text.
3. **Anomaly layer** — Isolation Forest, a feedforward Autoencoder, and an LSTM
   Autoencoder score URL features.
4. **Fusion layer** — combines both signals into a single classification
   (phishing / suspicious / legitimate).

The nine semantic × anomaly combinations form a 3 × 3 experimental matrix, each
evaluated against individual-model and traditional baselines.

---

## Repository structure

```
phishing-detection-hybrid/
├── README.md               # This file
├── NOTICE                  # Copyright (all rights reserved) + dual-use notice
├── requirements.txt        # Pinned dependencies
├── .gitignore              # Data policy enforced here
├── data/
│   ├── raw/                # Raw datasets (gitignored — obtain from source)
│   ├── processed/          # Numerical artefacts only (scaler, schema)
│   └── README.md           # Data policy + sources + regeneration steps
├── src/
│   ├── preprocessing/      # Corpus preparation pipelines (G1–G3)
│   ├── models/             # Semantic, anomaly, fusion models (in progress)
│   └── evaluation/         # Benchmarking + comparison matrix (in progress)
├── reports/
│   └── readiness/          # JSON data-readiness reports (per corpus)
├── docs/                   # Summaries, LSEPI/ethics, design documentation
├── notebooks/              # Exploration + evaluation notebooks
└── results/                # Metrics, figures, comparison matrices
```

---

## Reproducing the data preparation

Raw data is not distributed with this repository (see `data/README.md`). With the
raw datasets placed in `data/raw/`:

```bash
# G1 — semantic training corpus (email body text)
python src/preprocessing/preprocess_semantic_corpus.py data/raw/<file>.csv data/processed/

# G2 — anomaly training corpus (URL features)
python src/preprocessing/preprocess_anomaly_corpus.py data/raw/Prasad.csv data/processed/

# G3 — joint evaluation corpus (body text + URLs)
python src/preprocessing/preprocess_joint_corpus.py \
    data/raw/CEAS_08.csv data/processed/standard_scaler.pkl data/processed/ --name ceas
```

Every pipeline emits a JSON readiness report (in `reports/readiness/`) recording
all transformations, exclusions, and final distributions.

---

## Project status

| Phase | Status |
|-------|--------|
| Data preparation (G1–G3) | Complete (Presidio NER pass pending) |
| Semantic layer (G4) | Not started |
| Anomaly layer (G5) | Not started |
| Fusion layer (G6) | Not started |
| Evaluation (G7–G9) | Not started |
| Documentation (G10–G11) | Ongoing |

---

## Ethics and responsible use

This project is **protective research only**. Raw email data is never published,
trained model weights are deliberately withheld to prevent dual-use. See `NOTICE` and `docs/` for the full LSEPI, ethics,
and data-protection documentation.
