# Data Directory

This directory holds datasets and processed artefacts. **Most data is deliberately
excluded from version control** — see the policy below and the root `.gitignore`.

## Layout

```
data/
├── raw/         # Raw datasets — NEVER committed. Obtain from sources below.
└── processed/   # Processed artefacts. Only NUMERICAL, non-email-content
                 # files are version-controlled (scaler, feature schema).
```

## Data policy

| Data type | In version control? | Why |
|-----------|--------------------|-----|
| Raw datasets (email corpora, URL datasets) | **No** | Licensing, size, and PII risk. Reproduced from source. |
| Processed text corpora (semantic, joint text table) | **No** | Contain redacted email body text; residual free-text PII until the Presidio NER pass is complete. |
| Processed URL-feature data (numerical) | Optional | No email content. May be committed or regenerated from scripts. |
| Fitted scaler, feature schema | **Yes** | Small, no email content, required for reproducibility. |
| JSON readiness reports | **Yes** (in `reports/`) | Documentation only; no email content. |
| Trained model weights | **No** | Dual-use risk (see `NOTICE`) and size. |

## Obtaining the raw data

The raw datasets are publicly available from their original sources. Place them
in `data/raw/` before running the preprocessing scripts. (Record exact source
URLs and access dates here as you collect them — required for the methodology.)

| Dataset | Role | Source |
|---------|------|--------|
| Chakraborty_S | Semantic training (G1) | _record source + access date_ |
| Kuladeep | Semantic training (G1) | _record source + access date_ |
| Engineering_Informatica_Spa | Semantic training (G1) | _record source + access date_ |
| PhiUSIIL (Prasad) | Anomaly training (G2) | UCI ML Repository (CC BY 4.0) |
| CEAS 2008 | Joint evaluation (G3) | _record source + access date_ |
| TREC 2007 | Secondary evaluation (G3) | _record source + access date_ |

## Regenerating processed artefacts

From the repository root, with raw data in `data/raw/`:

```bash
# G1 — semantic corpus
python src/preprocessing/preprocess_semantic_corpus.py data/raw/<file>.csv data/processed/

# G2 — anomaly corpus
python src/preprocessing/preprocess_anomaly_corpus.py data/raw/Prasad.csv data/processed/

# G3 — joint evaluation corpus
python src/preprocessing/preprocess_joint_corpus.py \
    data/raw/CEAS_08.csv data/processed/standard_scaler.pkl data/processed/ --name ceas
```
