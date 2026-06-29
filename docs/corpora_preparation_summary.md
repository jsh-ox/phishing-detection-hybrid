# Corpora Preparation — Summary for Supervisor

**Project:** Hybrid ML for phishing detection in emails
**Scope of this summary:** Data preparation goals G1–G3 (semantic, anomaly, and joint-evaluation corpora)
**Status:** All three corpora built, validated, and version-controlled. One planned second-pass step (Presidio NER) outstanding.

---

## 1. Work completed

Three corpora were prepared, each through a documented, reusable Python pipeline that emits a cleaned data artefact plus a JSON readiness report recording every transformation.

**G1 — Semantic training corpus (email body text).**
Combined three sources — Chakraborty_S (17,483 rows), Kuladeep (9,956), and Engineering_Informatica_Spa (609) — into a single corpus of **27,986 rows** after cleaning and cross-source deduplication. Class balance is healthy at ~54% safe / ~46% phishing, so no resampling was required. Body text was whitespace-normalised and passed through a regex PII redaction pass (email addresses, phone numbers, card- and SSN-like numbers, and URLs replaced with tokens). Each row is tagged with its source dataset for provenance.

**G2 — Anomaly training corpus (URL features).**
PhiUSIIL (the "Prasad" file) was selected as the sole source. The corpus was restricted to **18 URL-string-derived features** and split 80/10/10 into train (188,296), validation (23,537), and test (23,537), with a legitimate-only subset (80,416) reserved for autoencoder training. A `StandardScaler` was fitted on the training split only and saved as an artefact for reuse, preventing data leakage.

**G3 — Joint evaluation corpus (body text + URLs).**
Built from CEAS 2008 (39,154 emails). Produces two linked tables: a text table (one row per email, cleaned and redacted identically to G1) and a URL-features table (145,539 rows — 133,078 real URLs plus 12,461 null-URL placeholders), with URL features computed from each email's embedded links and scaled using the saved G2 scaler. TREC 2007 is held back as a secondary evaluation set.

---

## 2. Key assumptions

- **Label mapping (Engineering_Informatica_Spa):** this source used fine-grained attack categories (Phishing, Scareware, Baiting, Malware, Pretexting). These were mapped to binary as *phishing*, with only "NOT-Malicious" mapped to *safe*.
- **PII redaction preserves signal:** replacing identifiers with tokens (e.g. `<URL>`, `<EMAIL>`) retains the structural cue that an identifier was present without keeping the identifier itself.
- **Null-URL representation:** emails containing no URL are represented by a neutral zero vector in scaled feature space, treated as "no URL signal" rather than an anomalous URL.
- **Multi-URL emails:** all URLs are retained as separate rows; a single anomaly score per email will be obtained by taking the maximum across its URLs at inference.

---

## 3. Limitations (flagged for the dissertation)

- **PII handling is first-pass only.** Regex redaction reliably catches structured identifiers but not free-text names. A Microsoft Presidio NER second pass is planned and not yet applied.
- **Lost per-source provenance.** Chakraborty_S is a pre-assembled CSV, so the original underlying sources (Nazario, Enron, etc.) cannot be individually attributed within it.
- **Synthetic data in the mix.** The Kuladeep source appears to be synthetically generated (templated signatures, appended keyword lists); this may affect how well the semantic models generalise to real email.
- **URL feature parity is approximate (~95%).** For G3, URL features are produced by a reimplementation of the PhiUSIIL feature definitions. The original extraction applied undocumented conventions (e.g. trimming a trailing character, stripping `www`) that were reverse-engineered by matching against PhiUSIIL's recorded values. Residual differences (~5%) introduce a minor train/serve skew, accepted and documented rather than eliminated.

---

## 4. Changes from the original plan

- **Semantic sources differ from the proposal.** The plan named Nazario + Enron + Nigerian Fraud 419; the corpus was instead built from the datasets actually available (Chakraborty_S, Kuladeep, Engineering_Informatica_Spa). The pipeline is source-agnostic, so others can be folded in later.
- **Anomaly corpus narrowed to one source.** The plan named PhiUSIIL + ISCX-URL-2016; in practice the candidate URL datasets (Vrbancic, Rachana, Jishnu_1/2) had incompatible feature schemas or only a single class, so **PhiUSIIL was used alone**. The others were set aside with documented reasons.
- **Feature count reduced 21 → 18.** Three reference-dependent features (CharContinuationRate, TLDLegitimateProb, URLCharProb) were removed because they cannot be reproduced for email URLs at evaluation without PhiUSIIL's internal probability tables. The label-leaking `URLSimilarityIndex` was also excluded.
- **Deduplication key corrected.** After removing those three features, deduplicating on the feature vector was found to collapse ~86% of genuinely distinct URLs; the pipeline now deduplicates on the raw URL string, preserving the dataset and its class balance.
- **URLs redacted in semantic body text.** The original plan suggested retaining URLs in body text for linguistic signal; in practice they were redacted to a `<URL>` token for consistency with PII handling, with the structural cue preserved.

---

## 5. Outstanding before model training

- Apply the planned Presidio NER second pass for free-text names across the text corpora.
- Process TREC 2007 through the same joint pipeline as the secondary evaluation set (when required).
- Tokenisation and any model-specific formatting for the semantic layer.

*All pipelines, artefacts, and readiness reports are reproducible and version-controlled, supporting the reproducibility goal (G10).*
