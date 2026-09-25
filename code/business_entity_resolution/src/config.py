"""
Configuration settings and paths for Amazon ML Challenge 2026.
Supports local execution and Google Colab execution seamlessly.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.getenv("AMAZON_ML_BASE_DIR", Path(__file__).resolve().parent.parent))

DATA_DIR = Path(os.getenv("AMAZON_ML_DATA_DIR", BASE_DIR / "dataset"))
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"

MODELS_DIR = Path(os.getenv("AMAZON_ML_MODELS_DIR", BASE_DIR / "models"))
OUTPUT_DIR = Path(os.getenv("AMAZON_ML_OUTPUT_DIR", BASE_DIR / "output"))

# Ensure writable directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file paths
TRAIN_S1_PATH = TRAIN_DATA_DIR / "train_source1.tsv"
TRAIN_S2_PATH = TRAIN_DATA_DIR / "train_source2.tsv"
TRAIN_S3_PATH = TRAIN_DATA_DIR / "train_source3.tsv"
TRAIN_GT_PATH = TRAIN_DATA_DIR / "train_ground_truth.tsv"

TEST_S1_PATH = TEST_DATA_DIR / "test_source1.tsv"
TEST_S2_PATH = TEST_DATA_DIR / "test_source2.tsv"
TEST_S3_PATH = TEST_DATA_DIR / "test_source3.tsv"

# Output submission paths
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Expected column schemas
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCHING_RESULTS_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_PAIRS_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]

# Multi-pass Blocking Hyperparameters
BLOCKING_CONFIG = {
    # Token-based blocking
    "min_name_token_len": 3,
    "min_addr_token_len": 4,
    "max_token_frequency": 5000,  # avoid exploding inverted index for hyper-frequent stopwords
    
    # Character TF-IDF parameters
    "tfidf_ngram_range": (3, 4),
    "tfidf_min_df": 2,
    "tfidf_max_features": 100_000,
    
    # TF-IDF cosine similarity retrieval
    "tfidf_name_top_k": 5,
    "tfidf_name_threshold": 0.45,
    "tfidf_addr_top_k": 5,
    "tfidf_addr_threshold": 0.40,
    
    # Max candidates per S1 entity
    "max_candidates_per_s1": 50,
}

# Evaluation settings
F_BETA = 0.5
RANDOM_SEED = 42
