# Amazon ML Challenge 2026: Business Entity Resolution

This repository contains the end-to-end machine learning pipeline developed for the **Amazon ML Challenge 2026 Business Entity Resolution Challenge**.

---

## 1. Problem Overview

In large-scale commercial platforms, business identity data arrives from multiple independent sources with noisy, incomplete, and inconsistent records. The task is to link reference entities from **Source 1** (`S1-*`) to their zero, one, or multiple corresponding matching entities in **Source 2** (`S2-*`) and **Source 3** (`S3-*`).

- **Reference Source**: Source 1 (deduplicated).
- **Matching Targets**: Source 2 and Source 3.
- **Record Schema**: `entity_id`, `business_name`, `business_address`, `country`.
- **Target Outputs**:
  - `matching_results.tsv`: Final predicted links (scored via macro $F_{0.5}$).
  - `candidate_pairs.tsv`: Final candidate set produced by blocking before model scoring.
- **Constraints**:
  - Strictly **NO external lookups**, no external APIs, no geocoding, and no pre-trained commercial business databases.
  - Country is an **open-set** string field (training contains `US` and `India`; test contains `US`, `India`, and `France`).
  - Open-source, permissive license compliant (XGBoost under Apache 2.0).

---

## 2. Project Directory Structure

```text
amazon_ml_entity_resolution/
│
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralized paths and hyperparameters
│   ├── data_loader.py         # Strict TSV loading, validation, and export
│   ├── preprocessing.py       # Open-set name and address text normalization
│   ├── blocking.py            # 5-pass candidate generation (exact, tokens, TF-IDF)
│   ├── features.py            # 36 pairwise similarity features
│   ├── train_model.py         # XGBoost classifier training & threshold search
│   ├── evaluate.py            # Complete validation evaluation & metrics
│   ├── inference.py           # Candidate generation & matching inference runner
│   └── metrics.py             # Official Macro F_0.5 and candidate recall metrics
│
├── models/
│   ├── matcher.pkl            # Serialized XGBoost model + best threshold
│   └── metadata.json          # Model metadata, parameters, and validation metrics
│
├── output/                    # Generated candidate_pairs.tsv and matching_results.tsv
├── requirements.txt           # Pinned dependencies
├── README.md                  # Complete documentation and Colab guide
└── Documentation_template.md  # Official challenge methodology template
```

---

## 3. Dataset Characteristics (from Exploratory Data Analysis)

### File Row Counts
- **Training Set**:
  - `train_source1.tsv`: 2,206,821 records
  - `train_source2.tsv`: 5,034,616 records
  - `train_source3.tsv`: 5,285,603 records
  - `train_ground_truth.tsv`: 2,206,821 records
  - *Total Training Records*: 12,527,040 rows
- **Test Set**:
  - `test_source1.tsv`: 1,732,544 records
  - `test_source2.tsv`: 4,887,273 records
  - `test_source3.tsv`: 5,082,316 records
  - *Total Test Records*: 11,702,133 rows

### Missing Value Analysis
- `entity_id`: 0 missing across all files.
- `business_name`: 0 missing across all files.
- `country`: 0 missing across all files.
- `business_address`:
  - `train_source1.tsv`: 0 (0.00%)
  - `train_source2.tsv`: 168,967 missing (3.36%)
  - `train_source3.tsv`: 175,916 missing (3.33%)
  - `test_source1.tsv`: 0 (0.00%)
  - `test_source2.tsv`: 129,408 missing (2.65%)
  - `test_source3.tsv`: 136,098 missing (2.68%)

### Country Distribution (Open-Set)
- **Training**:
  - `US`: ~60.0%
  - `India`: ~40.0%
- **Test**:
  - `India`: ~47.0%
  - `US`: ~38.3%
  - `France`: ~14.7% (Unseen country handled dynamically by open-set country partitioning).

### Ground Truth Link Topology
- **Singletons (0 matches)**: 123,247 records (5.58%)
- **Single match (1 match)**: 119,157 records (5.40%)
- **Multiple matches (>1 matches)**: 1,964,417 records (89.02%)
- **Max matches for a single S1 record**: 11 matches
- **Total true pairwise links**: 7,638,365 links
- **Mean matches per S1**: 3.4613

---

## 4. Multi-Pass Candidate Generation (Blocking)

The blocking pipeline implements 5 complementary passes, dynamically partitioned by country:

1. **Pass 1 — Exact Normalized Name**: Inverted index matching exact cleaned business names (handles minor legal suffix differences like "Inc", "LLC", "SARL").
2. **Pass 2 — Name Token & Prefix Blocking**: Inverted index on significant name tokens (length $\ge 3$) and distinctive prefixes, with frequency caps to prevent stopword explosions.
3. **Pass 3 — Address Number & Token Blocking**: Co-occurrence indexing of street/building numbers and distinctive address tokens.
4. **Pass 4 — Character TF-IDF Cosine Retrieval (Names)**: Sublinear TF-IDF over 3-4 char n-grams with sparse cosine dot-product retrieval for typo tolerance (e.g. `Wilblims` vs `Williams`).
5. **Pass 5 — Character TF-IDF Cosine Retrieval (Addresses)**: Character n-gram similarity over non-empty normalized addresses to retrieve records with matching location but altered/trade names.

---

## 5. Pairwise Feature Engineering

The feature engineering module ([src/features.py](file:///d:/amzaon%20ml/amazon_ml_entity_resolution/src/features.py)) extracts 36 numerical similarity features across 3 categories:

### A. Business Name Features (14 features)
- `name_exact_orig`: Binary exact match on raw business names.
- `name_exact_norm`: Binary exact match on normalized names.
- `name_token_jaccard`: Jaccard similarity of normalized word tokens.
- `name_token_overlap`: Token overlap coefficient $\frac{|A \cap B|}{\min(|A|, |B|)}$.
- `name_token_count_diff`: Absolute difference in token counts.
- `name_len_diff`: Absolute character length difference.
- `name_len_ratio`: Character length ratio $\frac{\min(L_1, L_2)}{\max(L_1, L_2)}$.
- `name_levenshtein_sim`: Normalized Levenshtein similarity via RapidFuzz.
- `name_jaro_winkler`: Jaro-Winkler string similarity.
- `name_fuzz_ratio`: RapidFuzz ratio / 100.
- `name_fuzz_token_sort_ratio`: Word-order invariant token sort ratio.
- `name_fuzz_token_set_ratio`: Subset/superset token set ratio.
- `name_char_3gram_jaccard`: Character 3-gram Jaccard similarity.
- `name_char_cosine`: Character 3-gram cosine similarity.

### B. Business Address Features (18 features)
- `addr_is_missing`: Flag indicating if either address is empty.
- `addr_both_missing`: Flag indicating if both addresses are empty.
- `addr_exact_orig`: Exact match on raw address strings.
- `addr_exact_norm`: Exact match on normalized address strings.
- `addr_token_jaccard`: Address token Jaccard similarity.
- `addr_token_overlap`: Address token overlap coefficient.
- `addr_token_count_diff`: Difference in address token count.
- `addr_len_diff`: Absolute character length difference.
- `addr_len_ratio`: Character length ratio.
- `addr_levenshtein_sim`: Address normalized Levenshtein similarity.
- `addr_jaro_winkler`: Address Jaro-Winkler similarity.
- `addr_fuzz_ratio`: Address RapidFuzz ratio / 100.
- `addr_fuzz_token_sort_ratio`: Address token sort ratio.
- `addr_fuzz_token_set_ratio`: Address token set ratio.
- `addr_char_3gram_jaccard`: Character 3-gram Jaccard similarity.
- `addr_char_cosine`: Character 3-gram cosine similarity.
- `addr_num_exact_match`: Equality of extracted address numbers / PIN codes.
- `addr_num_jaccard`: Jaccard similarity of extracted number sequences.

### C. Cross-Field & Source Features (4 features)
- `country_exact_match`: Binary equality of normalized country strings.
- `name_addr_sim_mean`: Mean of name and address token sort similarity.
- `name_addr_sim_mult`: Non-linear interaction product of name and address similarity.
- `candidate_source`: Indicator for candidate source origin (`2.0` for S2, `3.0` for S3).

---

## 6. Model Architecture & Validation Results

### XGBoost Classifier Configuration
```json
{
  "n_estimators": 250,
  "max_depth": 5,
  "learning_rate": 0.08,
  "subsample": 0.85,
  "colsample_bytree": 0.85,
  "scale_pos_weight": 5.0,
  "eval_metric": "logloss",
  "random_state": 42
}
```

### Validation Methodology & Performance
- **Validation Split**: Grouped strictly at the **Source 1 entity level** (80% train, 20% validation) to prevent pair leakage.
- **Optimal Probability Threshold**: **0.47** (selected via grid search strictly on validation macro $F_{0.5}$).
- **Validation Macro $F_{0.5}$**: **0.7850** (primary challenge evaluation metric).
- **Validation Precision**: **99.44%** (precision-heavy matching).
- **Validation Recall**: **68.15%**.
- **Singleton Accuracy**: **98.65%** (correctly predicts empty lists for singletons).
- **Model Artifact**: Serialized to `models/matcher.pkl` alongside `models/metadata.json`.

---

## 7. How to Reproduce Training and Evaluation

### Train Matcher & Optimize Threshold
```bash
python -m src.train_model --samples 15000 --val-split 0.20
```

### Run Comprehensive Evaluation
```bash
python -m src.evaluate --samples 2500
```

### Run Test Inference
```bash
python -m src.inference --full-inference
```
This generates both:
- `output/candidate_pairs.tsv` (blocking candidate set)
- `output/matching_results.tsv` (final matches for leaderboard submission)

### Validate Output Formatting
```bash
python /path/to/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 8. How to Run in Google Colab (Step-by-Step)

### Step 1: Upload or Clone the Repository to Google Drive
Place `amazon_ml_entity_resolution/` in Google Drive (e.g. `/content/drive/MyDrive/amazon_ml_entity_resolution`).

### Step 2: Open Colab and Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/amazon_ml_entity_resolution')
print("Current directory:", os.getcwd())
```

### Step 3: Install Dependencies
```python
!pip install -r requirements.txt
```

### Step 4: Train Matcher
```python
!python -m src.train_model --samples 15000 --val-split 0.20
```

### Step 5: Evaluate Validation Metrics
```python
!python -m src.evaluate --samples 2500
```

### Step 6: Generate Submission Files
```python
!python -m src.inference --full-inference
```
