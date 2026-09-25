# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Team Lead & Machine Learning Engineers  
**Submission Date:** September 2026

---

## 1. Executive Summary

We developed an open-set, highly scalable, and precision-optimized Entity Resolution (ER) pipeline for linking business records across three disparate sources without common identifiers. The architecture combines a high-recall multi-pass candidate generation system (using exact normalized name matching, token inverted indexing, numeric address co-occurrence, and character n-gram TF-IDF cosine retrieval) with an Apache 2.0-compliant XGBoost gradient-boosted decision tree classifier trained on 36 pairwise lexical, character, token, and numeric features. The decision threshold was strictly calibrated on held-out validation data to maximize the competition's macro-averaged $F_{0.5}$ metric, achieving a validation macro $F_{0.5}$ of **0.7850**, precision of **99.44%**, and singleton identification accuracy of **98.65%**, while remaining strictly compliant with the zero-external-lookup constraint.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across 12,527,040 training records and 11,702,133 test records revealed critical structural characteristics of real-world enterprise entity resolution:
- **Topology of Links**: In the ground truth, true matches follow an asymmetric distribution: 5.58% of Source 1 entities are singletons (0 matches), 5.40% have exactly 1 match, and 89.02% have multiple matches (up to 11 matches per entity), averaging 3.46 links per record. A one-to-one matching constraint would severely harm performance.
- **Open-Set Country Distribution**: While the training data spans only the United States (59.98%) and India (40.02%), the test set introduces a third country, **France** (accounting for 14.98% of Source 1, 14.39% of Source 2, and 14.40% of Source 3). The pipeline treats country as a dynamic open-set string label, avoiding any hardcoded country priors.
- **Noise Patterns**: Records exhibit diverse lexical corruptions:
  - *Legal Suffix Differences*: e.g., "Corp", "Inc", "LLC", "Pvt Ltd", "SARL", "SAS" frequently appear or disappear across sources.
  - *Address Omissions*: 3.36% of Source 2 and 3.33% of Source 3 records completely lack addresses (or contain literal `'null'` strings).
  - *Address Formatting & Numbering*: Street abbreviations ("Ave" vs "Avenue", "Rd" vs "Road"), ordinal typos ("45nd" vs "45th"), and component permutations are prevalent.
  - *Indic Scripts & Transliteration*: Indian records exhibit English text, native scripts (e.g., Tamil, Devanagari), and mixed domain names (e.g., `.com` or `.in` appendages).

### 2.2 Solution Strategy
We formulated entity resolution as a two-stage machine learning system:
$$\text{Test Records} \xrightarrow{\text{Normalization}} \text{Multi-Pass Blocking} \xrightarrow{\text{Candidate Set}} \text{Pairwise Feature Extraction} \xrightarrow{\text{XGBoost Scoring}} \text{Threshold Calibration} \xrightarrow{} \text{Final Matches}$$

- **Approach Type**: Multi-Pass Blocking + Gradient Boosted Decision Tree (XGBoost) Pairwise Classifier.
- **Core Innovation**:
  1. *Dynamic Open-Set Country Partitioning*: Blocking is partitioned strictly within identical country labels, drastically reducing the search space from $O(N \cdot M)$ without hardcoding country sets.
  2. *Noise-Resilient Text Normalization*: Custom regex engine handling Unicode NFKD normalization, multi-language legal suffix stripping (US, India, and France), URL/domain stripping, and street ordinal standardization.
  3. *Precision-Heavy Threshold Optimization*: Calibrated decision threshold ($\tau^* = 0.470$) tailored for the macro $F_{0.5}$ metric, prioritizing false merge avoidance over marginal recall gains.
  4. *Low-Memory Streaming Architecture*: Multi-pass candidate indexing and prediction designed to run in memory-constrained environments (< 1 GB RAM).

---

## 3. Candidate Generation (Blocking)

To reduce the comparison space from over $2.2 \times 10^{13}$ possible pairs to a computationally tractable candidate pool without losing true matches, we developed a 5-pass blocking system:

- **Blocking Strategies Used**:
  - **Pass 1 (Exact Normalized Name)**: Inverted index mapping cleaned business names stripped of legal suffixes and punctuation.
  - **Pass 2 (Name Token & Prefix Inverted Index)**: Indexes distinctive word tokens ($\ge 4$ characters) with inverted index frequency caps ($\le 1,000$) to prevent stopword explosion.
  - **Pass 3 (Address Number & Token Co-occurrence)**: Co-occurrence index linking entities that share a numeric component (house/building/PIN code) and an address token.
  - **Pass 4 (Character n-gram TF-IDF for Names)**: Sublinear 3-to-4 character n-gram cosine retrieval using sparse matrix multiplications to catch typographical errors (e.g., "Wilblims" vs "Williams").
  - **Pass 5 (Character n-gram TF-IDF for Addresses)**: Character n-gram similarity on non-empty addresses to link businesses with identical locations but trade/DBA name variations.
- **Candidate Deduplication**: All 5 passes are combined via set union and deduplicated per Source 1 entity, yielding a **59.35%** deduplication reduction.
- **Candidate Recall Ceiling**: On validation ground truth evaluations, the blocking stage captured **84.27%** of all true pairwise links and achieved an **Entity Partial Recall of 93.49%**, with an average of only **28.95 candidates per Source 1 entity**.

---

## 4. Matching Model

### 4.1 Features Used (36 Numerical Features)
The feature engineering module ([src/features.py](file:///d:/amzaon%20ml/amazon_ml_entity_resolution/src/features.py)) extracts 36 pairwise similarity signals:

- **Name Features (14)**:
  - Exact match on original and normalized names (`name_exact_orig`, `name_exact_norm`).
  - Word token Jaccard similarity and token overlap coefficient (`name_token_jaccard`, `name_token_overlap`).
  - Word token count difference and character length difference/ratio (`name_token_count_diff`, `name_len_diff`, `name_len_ratio`).
  - Edit distance metrics: Normalized Levenshtein similarity, Jaro-Winkler distance, RapidFuzz ratio, token sort ratio, token set ratio (`name_levenshtein_sim`, `name_jaro_winkler`, `name_fuzz_ratio`, `name_fuzz_token_sort_ratio`, `name_fuzz_token_set_ratio`).
  - Character n-gram Jaccard and cosine similarity (`name_char_3gram_jaccard`, `name_char_cosine`).
- **Address Features (18)**:
  - Missingness indicators (`addr_is_missing`, `addr_both_missing`).
  - Exact match on original and normalized addresses (`addr_exact_orig`, `addr_exact_norm`).
  - Address token Jaccard, token overlap, and token count difference (`addr_token_jaccard`, `addr_token_overlap`, `addr_token_count_diff`).
  - Address length difference and ratio (`addr_len_diff`, `addr_len_ratio`).
  - RapidFuzz normalized Levenshtein, Jaro-Winkler, token sort ratio, and token set ratio (`addr_levenshtein_sim`, `addr_jaro_winkler`, `addr_fuzz_ratio`, `addr_fuzz_token_sort_ratio`, `addr_fuzz_token_set_ratio`).
  - Address character 3-gram Jaccard and cosine similarity (`addr_char_3gram_jaccard`, `addr_char_cosine`).
  - Numeric address equality and Jaccard similarity on extracted street/postal digits (`addr_num_exact_match`, `addr_num_jaccard`).
- **Cross-Field Features (4)**:
  - `country_exact_match`: Binary flag for identical country label.
  - `name_addr_sim_mean`: Arithmetic mean of name and address similarity.
  - `name_addr_sim_mult`: Non-linear interaction product of name and address similarities.
  - `candidate_source`: Numerical source indicator (2.0 for S2, 3.0 for S3).

### 4.2 Model Architecture & Training
- **Model Type**: XGBoost Classifier (`XGBClassifier`) under Apache 2.0 license (< 8 Billion parameters).
- **Hyperparameters**:
  - `n_estimators`: 250
  - `max_depth`: 5
  - `learning_rate`: 0.08
  - `subsample`: 0.85
  - `colsample_bytree`: 0.85
  - `scale_pos_weight`: 5.0 (calibrated against negative-to-positive candidate imbalance)
  - `eval_metric`: `"logloss"` with early stopping rounds = 25
  - `random_state`: 42
- **Validation Methodology**: Grouped strictly at the **`source1_entity_id` level** (80% train / 20% validation) to eliminate pair-level data leakage.
- **Threshold Selection Method**: Grid search across $\tau \in [0.15, 0.90]$ with step $0.02$ optimizing the official macro $F_{0.5}$ metric across all validation Source 1 entities (including singletons). The optimal threshold selected was $\mathbf{\tau^* = 0.470}$.

---

## 5. Results & Error Analysis

### 5.1 Validation Performance Metrics
On held-out validation Source 1 entities:
- **Macro $F_{0.5}$ Score**: **0.7850** (Official Challenge Scored Metric)
- **Overall Precision**: **99.44%** (3,554 True Positives, 20 False Positives)
- **Overall Recall**: **68.15%**
- **Singleton Accuracy ($F_{0.5}$)**: **98.65%** (73 out of 74 singletons correctly predicted as empty)
- **Non-Singleton Macro $F_{0.5}$**: **0.7746**

### 5.2 Error Analysis
- **Common False Positives (Wrong Merges)**: Almost entirely eliminated by the high precision threshold ($\tau^* = 0.470$). The few observed false merges occurred between co-located franchised businesses sharing identical street addresses and similar parent brand names (e.g. generic trade names in the same commercial complex).
- **Common False Negatives (Missed Matches)**: Primarily caused by records exhibiting extreme abbreviation discrepancies combined with completely omitted address fields (where both address fields were empty, leaving insufficient overlap signals for high-confidence classification).

---

## 6. Conclusion
The entity resolution pipeline demonstrates that combining domain-specific multi-pass blocking with conservative, highly regularized gradient boosting on pairwise similarity features achieves superior performance on large-scale noisy business data. By optimizing specifically for macro $F_{0.5}$ and explicitly penalizing false merges on singletons, the solution achieves near-perfect precision (99.44%) while maintaining high macro entity resolution quality across open-set international boundaries.

---

## Appendix

### A. Code Artefacts
The complete runnable source code is organized under `code/business_entity_resolution/`:
```text
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration & hyperparameters
│   ├── data_loader.py         # TSV validation and I/O
│   ├── preprocessing.py       # Open-set text normalizer
│   ├── blocking.py            # 5-pass candidate generation
│   ├── features.py            # 36 pairwise similarity features
│   ├── train_model.py         # XGBoost training & threshold optimization
│   ├── evaluate.py            # Comprehensive validation evaluation
│   ├── inference.py           # Streaming test inference runner
│   └── metrics.py             # Macro F0.5 & blocking metrics
├── README.md                  # Complete reproduction & Colab guide
└── requirements.txt           # Pinned dependencies
```

**Entry Points to Reproduce**:
- **Train Model & Optimize Threshold**:
  ```bash
  python -m src.train_model --samples 15000 --val-split 0.20
  ```
- **Evaluate Validation Performance**:
  ```bash
  python -m src.evaluate --samples 2500
  ```
- **Generate Final Submissions**:
  ```bash
  python -m src.inference
  ```
  Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- **Validate Submission Integrity**:
  ```bash
  python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
  ```

### B. Additional Results
- Feature importance analysis revealed that address token set similarity (`addr_fuzz_token_set_ratio`, importance 0.8379), address token overlap (`addr_token_overlap`, 0.0426), and combined cross-field mean similarity (`name_addr_sim_mean`, 0.0341) provided the strongest discriminative signals for entity equivalence.
- The pipeline was verified with the official submission validator `utils/validate_submission.py`, achieving status `PASS` with zero schema violations, zero missing entities, and zero self-matches across all 1,732,544 test Source 1 entities.
