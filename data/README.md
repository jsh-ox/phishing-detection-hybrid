# Data Directory

This directory holds datasets and processed artefacts. **Data is deliberately
excluded from version control** — see the policy below and the root `.gitignore`.

## Layout

```
data/
├── raw/         # Raw datasets — gitignored. Obtain from sources below.
└── processed/   # Processed artefacts. Only NUMERICAL, non-email-content
                 # files are version-controlled (scaler, feature schema).
```

## Data policy

| Data type | In version control? | Why |
|-----------|--------------------|-----|
| Raw datasets (email corpora, URL datasets) | **No** | Size, and PII risk. Reproduced from source. |
| Processed text corpora (semantic, joint text table) | **No** | Contain redacted email body text; potential residual free-text PII until the Presidio NER pass is complete. |
| Processed URL-feature data (numerical) | Optional | No email content. May be committed or regenerated from scripts. |
| Fitted scaler, feature schema | **Yes** | Small, no email content, required for reproducibility. |
| JSON readiness reports | **Yes** (in `reports/`) | Documentation only; no email content. |
| Trained model weights | **No** | Dual-use risk (see `NOTICE`) and size. |

## Obtaining the raw data

The raw datasets are publicly available from their original sources. Place them
in `data/raw/` before running the preprocessing scripts.

| Dataset | Role | Source |
|---------|------|--------|
| Chakraborty_S | Semantic training (G1) | _Chakraborty, S. (2023) Phishing Email Detection [Online]. Available at https://www.kaggle.com/datasets/subhajournal/phishingemails (Accessed 30 March 2026)._ |
| Engineering_Informatica_Spa | Semantic training (G1) | _Engineering Ingegneria Informatica Spa (2025) ‘Multiclass NLP Dataset for Phishing and Social Engineering Threat Detection’, Zenodo [Online]. DOI: https://doi.org/10.5281/zenodo.15235123 (Accessed 30 March 2026)._ |
| PhiUSIIL | Anomaly training (G2) | _Prasad, A. and Chandra, S. (2024) ‘PhiUSIIL Phishing URL (Website)’, UCI Machine Learning Repository [Online]. DOI: https://doi.org/10.1016/j.cose.2023.103545 (Accessed 30 April 2026)._ |
| CEAS 2008 | Joint evaluation (G3) | _CEAS (2008) Conference on Email and Anti-Spam 2008 [Online]. Available at https://www.ceas.cc/ (Accessed 29 June 2026). rokibulroni (2025) GitHub - rokibulroni/Phishing-Email-Dataset [Online]. Available at https://github.com/rokibulroni/Phishing-Email-Dataset (Accessed 28 April 2026)._ |

## Regenerating processed artefacts

From the repository root, with supplimented raw data in `data/raw/`:

```bash
# G1 — semantic corpus
python src/preprocessing/preprocess_semantic_corpus.py data/raw/<file>.csv data/processed/

# G2 — anomaly corpus
python src/preprocessing/preprocess_anomaly_corpus.py data/raw/<file>.csv data/processed/

#G2 — LSTM Autoencoder
python src/preprocessing/prepare_url_sequences.py data/raw/<file>.csv data/processed/

# G3 — joint evaluation corpus
python src/preprocessing/preprocess_joint_corpus.py \
    data/raw/<file>.csv data/processed/standard_scaler.pkl data/processed/ --name ceas
```
