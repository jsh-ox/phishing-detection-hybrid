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

1. **Preprocessing** - shared cleaning, redaction, and feature extraction.
2. **Semantic layer** - BERT, RoBERTa, and Sentence-BERT analyse email body text.
3. **Anomaly layer** - Isolation Forest, a feedforward Autoencoder, and an LSTM
   Autoencoder score URL features.
4. **Fusion layer** - combines both signals into a single classification
   (phishing / suspicious / legitimate).

The nine semantic × anomaly combinations form a 3 × 3 matrix, each
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
│   ├── raw/                # Raw datasets (gitignored — see separate README)
│   ├── processed/          # Numerical artefacts only (scaler, schema)
│   └── README.md           # Data policy + sources + regeneration steps
├── src/
│   ├── preprocessing/      # Corpus preparation pipelines (G1–G3)
│   ├── models/             # Semantic, anomaly, fusion models
│   └── evaluation/         # Benchmarking + comparison matrix
├── reports/
│   └── readiness/          # JSON data-readiness reports (per corpus)
├── docs/                   # Summaries, LSEPI/ethics, designs and documentation
├── notebooks/              # Exploration + evaluation notebooks
└── results/                # Metrics, figures, comparison matrices
```

---

## Reproducing the data preparation

Raw data is not distributed with this repository (see `data/README.md`). With the
raw datasets in `data/raw/`:

Preparation script can be found in the `data/processed`, for full detail see the separate `README`.

Every pipeline emits a JSON readiness report (in `reports/readiness/`) recording
all transformations, exclusions, and final distributions.

---

## Project status

| Phase | Status |
|-------|--------|
| Data preparation (G1–G3) | Complete |
| Semantic layer (G4) | In-progress |
| Anomaly layer (G5) | Second Iteration |
| Fusion layer (G6) | Not started |
| Evaluation (G7–G9) | Not started |
| Documentation (G10–G11) | Ongoing |

---

## Ethics and responsible use

This project is **protective research only**. Raw email data is never published,
trained model weights are deliberately withheld to prevent dual-use. See `NOTICE` and `docs/` for the full LSEPI, ethics,
and data-protection documentation.
