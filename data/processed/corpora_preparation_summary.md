# Corpora Preparation Summary

**Project:** Hybrid ML for phishing detection in emails
**Status:** All three corpora built, validated, and version-controlled. (Presidio NER) second-pass outstanding. G1 needs to be split into train/test/val.

---

## 1. Work completed

Three corpora prepared, through a reusable Python pipeline that produces a cleaned data artefact plus a JSON readiness report recording every transformation.

**G1 — Semantic training corpus (email body text).**
Combined two sources — Chakraborty_S (17,483 rows) and Engineering_Informatica_Spa (609) — into a single corpus of **18,030 rows** after cleaning and cross-source deduplication. Class balance is healthy at 61.6% safe / 38.4% phishing, so no resampling was required. Body text was whitespace-normalised and passed through a regex PII redaction pass (email addresses, phone numbers, card- and SSN-like numbers, and URLs replaced with tokens). Each row is tagged with its source dataset.

**G2 — Anomaly training corpus (URL features).**
PhiUSIIL was selected as the sole source. The corpus was restricted to **18 URL-string-derived features** and split 80/10/10 into train (188,296), validation (23,537), and test (23,537), with a legitimate-only subset (80,416) reserved for autoencoder training. A `StandardScaler` was fitted on the training split only and saved as an artefact for reuse, preventing data leakage.

**G3 — Joint evaluation corpus (body text + URLs).**
Built from CEAS 2008 (39,154 emails). Produces two linked tables: a text table (one row per email, cleaned and redacted identically to G1) and a URL-features table (145,539 rows — 133,078 real URLs plus 12,461 null-URL placeholders), with URL features computed from each email's embedded links and scaled using the saved G2 scaler.

---

## 2. Key assumptions

- **Label mapping (Engineering_Informatica_Spa):** this source used fine-grained attack categories (Phishing, Scareware, Baiting, Malware, Pretexting). These were mapped to binary as *phishing*, with only "NOT-Malicious" mapped to *safe*.
- **PII redaction preserves signal:** replacing identifiers with tokens (e.g. `<URL>`, `<EMAIL>`) retains the structural cue that an identifier was present without keeping the identifier itself.
- **Null-URL representation:** emails containing no URL are represented by a neutral zero vector in scaled feature space, treated as "no URL signal".
- **Multi-URL emails:** all URLs are retained as separate rows; a single anomaly score per email will be obtained by taking the maximum.

---

## 3. Limitations

- **PII handling is first-pass only.** Regex redaction reliably catches structured identifiers but not free-text names. A Microsoft Presidio NER second pass is planned and not yet applied.
- **Lost per-source provenance.** Chakraborty S. is a pre-assembled CSV, so the original underlying sources cannot be individually attributed within it.
- **URL feature parity is approximate (~95%).** For G3, URL features are produced by a reimplementation of the PhiUSIIL feature definitions. The original extraction applied undocumented conventions (e.g. trimming a trailing character, stripping `www`) that were reverse-engineered by matching against PhiUSIIL's recorded values. Residual tiny differences introduce a minor train/serve skew.
- **Feature count reduced 21 → 18.** Three reference-dependent features (CharContinuationRate, TLDLegitimateProb, URLCharProb) were removed because they cannot be reproduced for email URLs at evaluation without PhiUSIIL's internal probability tables. `URLSimilarityIndex` was also excluded as it is a powerful deterministic indicator of phishing emails and would likely overpower model predictions.

---

## 5. Outstanding work

- Apply Presidio NER second pass for free-text names across the text corpora.
- Build train/test/val split for G1.
- Consider necessity of combining more datasets for larger corpora.
- Tokenisation and any model-specific formatting.

*All pipelines, artefacts, and readiness reports are reproducible and version-controlled.*
