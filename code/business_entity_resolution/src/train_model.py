"""
Model training and threshold optimization pipeline for Amazon ML Challenge 2026.
Trains an XGBoost binary classifier on pairwise candidate features, optimizes the
decision threshold on validation data for macro-averaged F_0.5, and saves the model artifact.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Add base directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TRAIN_S1_PATH,
    TRAIN_S2_PATH,
    TRAIN_S3_PATH,
    TRAIN_GT_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    F_BETA,
)
from src.data_loader import load_source_file, load_ground_truth
from src.preprocessing import preprocess_dataframe
from src.blocking import MultiPassBlocker
from src.features import extract_features_for_pairs, FEATURE_COLUMNS
from src.metrics import compute_macro_f05, evaluate_blocking_recall


def find_best_threshold(
    val_s1_ids: List[str],
    val_pairs: List[Tuple[str, str]],
    val_probs: np.ndarray,
    gt_map: Dict[str, Set[str]],
    beta: float = 0.5,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Search for the decision threshold that maximizes macro F_beta on validation S1 entities.
    
    Parameters
    ----------
    val_s1_ids : List of S1 entity IDs in the validation split (including singletons).
    val_pairs : List of (s1_id, cand_id) tuples in validation candidate set.
    val_probs : Predicted probabilities for each validation candidate pair.
    gt_map : Ground truth dictionary mapping s1_id -> set of true matched IDs.
    beta : Weight parameter (beta=0.5).
    """
    if thresholds is None:
        thresholds = np.arange(0.15, 0.90, 0.02)

    best_thresh = 0.5
    best_macro_f05 = -1.0
    best_details = {}

    # Organize predictions by s1_id for quick filtering across thresholds
    s1_to_pair_probs = {s1_id: [] for s1_id in val_s1_ids}
    for (s1_id, cand_id), prob in zip(val_pairs, val_probs):
        if s1_id in s1_to_pair_probs:
            s1_to_pair_probs[s1_id].append((cand_id, float(prob)))

    # Evaluate each threshold
    for thresh in thresholds:
        preds = {}
        for s1_id in val_s1_ids:
            cand_probs = s1_to_pair_probs[s1_id]
            matched_cands = {c for c, p in cand_probs if p >= thresh}
            preds[s1_id] = matched_cands

        val_gt_subset = {s1_id: gt_map.get(s1_id, set()) for s1_id in val_s1_ids}
        macro_score, details = compute_macro_f05(val_gt_subset, preds, beta=beta)

        if macro_score > best_macro_f05:
            best_macro_f05 = macro_score
            best_thresh = float(thresh)
            best_details = details

    # Compute overall precision and recall at the best threshold
    best_preds = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for s1_id in val_s1_ids:
        cand_probs = s1_to_pair_probs[s1_id]
        matched_cands = {c for c, p in cand_probs if p >= best_thresh}
        best_preds[s1_id] = matched_cands
        true_set = gt_map.get(s1_id, set())
        tp = len(matched_cands.intersection(true_set))
        fp = len(matched_cands - true_set)
        fn = len(true_set - matched_cands)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    best_details["best_threshold"] = best_thresh
    best_details["overall_precision"] = float(overall_precision)
    best_details["overall_recall"] = float(overall_recall)
    best_details["total_predicted_matches"] = total_tp + total_fp
    best_details["total_true_matches"] = total_tp + total_fn

    return best_thresh, best_macro_f05, best_details


def train_pipeline(
    n_sample_s1: int = 25000,
    val_split_ratio: float = 0.20,
    random_seed: int = RANDOM_SEED,
    model_output_path: Optional[Path] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Run complete training pipeline:
    1. Sample S1 entities and split into train and validation (at S1 entity level).
    2. Extract relevant target pool from Source 2 and Source 3.
    3. Run MultiPassBlocker to generate candidate pairs for train and validation.
    4. Construct pairwise ground truth labels (0 or 1).
    5. Extract 36 numerical similarity features.
    6. Train XGBoost binary classification model with early stopping.
    7. Optimize decision threshold for validation macro F_0.5.
    8. Save model and metadata artifact.
    """
    t0 = time.time()
    if model_output_path is None:
        model_output_path = MODELS_DIR / "matcher.pkl"

    print("==========================================")
    print("STEP 1: Loading Dataset Sample for Training")
    print("==========================================")
    # Check if benchmark cache can be used for fast reproducible local training
    cache_dir = MODELS_DIR / "benchmark_cache"
    using_cached_sample = False
    if use_cache and (cache_dir / "sample_s1.tsv").is_file() and (cache_dir / "sample_targets.tsv").is_file():
        print(f"Loading training data from benchmark cache: {cache_dir.resolve()}")
        df_gt, gt_map = load_ground_truth(cache_dir / "sample_gt.tsv")
        s1_df = pd.read_csv(cache_dir / "sample_s1.tsv", sep="\t", keep_default_na=False)
        targets_df = pd.read_csv(cache_dir / "sample_targets.tsv", sep="\t", keep_default_na=False)
        s2_df = targets_df[targets_df["entity_id"].str.startswith("S2-")].copy()
        s3_df = targets_df[targets_df["entity_id"].str.startswith("S3-")].copy()
        all_s1_ids = df_gt["source1_entity_id"].values
        n_total_s1 = len(all_s1_ids)
        using_cached_sample = True
    else:
        # Load ground truth from raw files
        df_gt, gt_map = load_ground_truth(TRAIN_GT_PATH, nrows=n_sample_s1)
        all_s1_ids = df_gt["source1_entity_id"].values
        n_total_s1 = len(all_s1_ids)

    # Stratified entity-level train / val split
    np.random.seed(random_seed)
    shuffled_s1 = np.random.permutation(all_s1_ids)
    n_val = int(n_total_s1 * val_split_ratio)
    val_s1_ids = list(shuffled_s1[:n_val])
    train_s1_ids = list(shuffled_s1[n_val:])
    val_s1_set = set(val_s1_ids)
    train_s1_set = set(train_s1_ids)

    print(f"Total S1 entities: {n_total_s1:,}")
    print(f"Train S1 entities: {len(train_s1_ids):,} ({(1 - val_split_ratio)*100:.1f}%)")
    print(f"Val S1 entities:   {len(val_s1_ids):,} ({val_split_ratio*100:.1f}%)")

    if not using_cached_sample:
        # Load S1 records
        all_s1_set = set(all_s1_ids)
        s1_list = []
        for chunk in pd.read_csv(TRAIN_S1_PATH, sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
            hit = chunk[chunk["entity_id"].isin(all_s1_set)]
            if len(hit) > 0:
                s1_list.append(hit)
            if sum(len(x) for x in s1_list) >= n_total_s1:
                break
        s1_df = pd.concat(s1_list, ignore_index=True).drop_duplicates(subset=["entity_id"])

        # Collect true targets
        all_true_target_ids = set()
        for sid in all_s1_ids:
            all_true_target_ids.update(gt_map.get(sid, set()))
        print(f"Identified {len(all_true_target_ids):,} true target links in ground truth.")

        # Load S2 and S3 pools capturing 100% of true targets for the S1 sample + background negatives
        s2_list = []
        s2_needed = {t for t in all_true_target_ids if t.startswith("S2-")}
        s2_extra = 25000
        for chunk in pd.read_csv(TRAIN_S2_PATH, sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
            hit = chunk[chunk["entity_id"].isin(s2_needed)]
            if len(hit) > 0:
                s2_list.append(hit)
                s2_needed -= set(hit["entity_id"])
            if s2_extra > 0:
                take = min(s2_extra, 5000)
                s2_list.append(chunk.head(take))
                s2_extra -= take
            if not s2_needed and s2_extra <= 0:
                break
        s2_df = pd.concat(s2_list, ignore_index=True).drop_duplicates(subset=["entity_id"])

        s3_list = []
        s3_needed = {t for t in all_true_target_ids if t.startswith("S3-")}
        s3_extra = 25000
        for chunk in pd.read_csv(TRAIN_S3_PATH, sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
            hit = chunk[chunk["entity_id"].isin(s3_needed)]
            if len(hit) > 0:
                s3_list.append(hit)
                s3_needed -= set(hit["entity_id"])
            if s3_extra > 0:
                take = min(s3_extra, 5000)
                s3_list.append(chunk.head(take))
                s3_extra -= take
            if not s3_needed and s3_extra <= 0:
                break
        s3_df = pd.concat(s3_list, ignore_index=True).drop_duplicates(subset=["entity_id"])

    print(f"Loaded {len(s1_df):,} Source 1 records.")
    print(f"Loaded Target Pool: S2 = {len(s2_df):,}, S3 = {len(s3_df):,}")

    print("\n==========================================")
    print("STEP 2: Text Normalization")
    print("==========================================")
    s1_norm = preprocess_dataframe(s1_df)
    s2_norm = preprocess_dataframe(s2_df)
    s3_norm = preprocess_dataframe(s3_df)

    # Build fast lookup dicts for feature extraction
    s1_dict = s1_norm.set_index("entity_id").to_dict("index")
    target_norm = pd.concat([s2_norm, s3_norm], ignore_index=True).drop_duplicates(subset=["entity_id"])
    target_dict = target_norm.set_index("entity_id").to_dict("index")

    print("\n==========================================")
    print("STEP 3: Multi-Pass Candidate Generation")
    print("==========================================")
    blocker = MultiPassBlocker()
    candidates, blocking_stats = blocker.generate_candidates(s1_norm, s2_norm, s3_norm, verbose=True)

    # Evaluate blocking recall on validation
    pool_ids = set(target_norm["entity_id"])
    val_gt_eval = {sid: {m for m in gt_map.get(sid, set()) if m in pool_ids} for sid in val_s1_ids}
    val_blocking_metrics = evaluate_blocking_recall(val_gt_eval, candidates)
    print(f"\nValidation Blocking Candidate Recall: {val_blocking_metrics['candidate_recall']*100:.2f}%")
    print(f"Validation Entity Partial Recall:    {val_blocking_metrics['entity_partial_recall']*100:.2f}%")
    print(f"Validation Candidates Generated:     {val_blocking_metrics['total_candidate_pairs']:,}")

    print("\n==========================================")
    print("STEP 4: Construct Pairwise Training/Validation Sets")
    print("==========================================")
    train_pairs = []
    train_labels = []
    val_pairs = []
    val_labels = []

    for s1_id in all_s1_ids:
        cands = candidates.get(s1_id, set())
        true_matches = gt_map.get(s1_id, set())
        is_val = s1_id in val_s1_set

        for cand_id in cands:
            label = 1 if cand_id in true_matches else 0
            if is_val:
                val_pairs.append((s1_id, cand_id))
                val_labels.append(label)
            else:
                train_pairs.append((s1_id, cand_id))
                train_labels.append(label)

    train_labels = np.array(train_labels, dtype=np.int32)
    val_labels = np.array(val_labels, dtype=np.int32)

    n_train_pos = int((train_labels == 1).sum())
    n_train_neg = int((train_labels == 0).sum())
    n_val_pos = int((val_labels == 1).sum())
    n_val_neg = int((val_labels == 0).sum())

    print(f"Train Pairs: {len(train_pairs):,} (Positives: {n_train_pos:,}, Negatives: {n_train_neg:,}, Pos%: {n_train_pos/(len(train_pairs)+1e-9)*100:.2f}%)")
    print(f"Val Pairs:   {len(val_pairs):,} (Positives: {n_val_pos:,}, Negatives: {n_val_neg:,}, Pos%: {n_val_pos/(len(val_pairs)+1e-9)*100:.2f}%)")

    print("\n==========================================")
    print("STEP 5: Extracting 36 Pairwise Features")
    print("==========================================")
    print(f"Extracting features for {len(train_pairs):,} train pairs...")
    X_train = extract_features_for_pairs(train_pairs, s1_dict, target_dict)
    print(f"Extracting features for {len(val_pairs):,} validation pairs...")
    X_val = extract_features_for_pairs(val_pairs, s1_dict, target_dict)

    print("\n==========================================")
    print("STEP 6: Training XGBoost Binary Classifier")
    print("==========================================")
    # Calculate scale_pos_weight conservatively
    imbalance_ratio = n_train_neg / (n_train_pos + 1e-9)
    # Use sqrt of imbalance ratio to provide boost without inflating false positives
    scale_pos_weight = float(np.clip(np.sqrt(imbalance_ratio), 1.0, 5.0))
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}. scale_pos_weight set to: {scale_pos_weight:.2f}")

    xgb_params = {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": random_seed,
        "n_jobs": -1,
        "early_stopping_rounds": 25,
    }
    print(f"Model Configuration:\n{json.dumps(xgb_params, indent=2)}")

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_train,
        train_labels,
        eval_set=[(X_val, val_labels)],
        verbose=50,
    )

    print("\n==========================================")
    print("STEP 7: Threshold Optimization for Macro F_0.5")
    print("==========================================")
    val_probs = model.predict_proba(X_val)[:, 1]
    best_thresh, best_macro_f05, details = find_best_threshold(
        val_s1_ids=val_s1_ids,
        val_pairs=val_pairs,
        val_probs=val_probs,
        gt_map=gt_map,
        beta=F_BETA,
    )

    print(f"Optimal Decision Threshold:    {best_thresh:.2f}")
    print(f"Validation Macro F_0.5:        {best_macro_f05:.4f}")
    print(f"Validation Precision:          {details['overall_precision']*100:.2f}%")
    print(f"Validation Recall:             {details['overall_recall']*100:.2f}%")
    print(f"Validation Singleton Score:    {details['singleton_f05']*100:.2f}% ({details['num_singletons']} singletons)")
    print(f"Validation Non-Singleton F0.5: {details['non_singleton_f05']:.4f} ({details['num_non_singletons']} non-singletons)")
    print(f"Predicted Matches:             {details['total_predicted_matches']:,}")
    print(f"True Matches:                  {details['total_true_matches']:,}")

    # Top feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(-importances)
    print("\nTop 10 Most Important Features:")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank}. {FEATURE_COLUMNS[idx]}: {importances[idx]:.4f}")

    print("\n==========================================")
    print("STEP 8: Saving Model Artifact and Metadata")
    print("==========================================")
    artifact = {
        "model": model,
        "best_threshold": best_thresh,
        "feature_columns": FEATURE_COLUMNS,
        "validation_metrics": details,
        "blocking_metrics": val_blocking_metrics,
        "xgb_params": xgb_params,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(artifact, model_output_path)
    print(f"Model artifact saved to: {model_output_path.resolve()}")

    # Save JSON metadata
    metadata = {
        "best_threshold": best_thresh,
        "validation_macro_f05": best_macro_f05,
        "precision": details["overall_precision"],
        "recall": details["overall_recall"],
        "singleton_f05": details["singleton_f05"],
        "num_singletons": details["num_singletons"],
        "num_non_singletons": details["num_non_singletons"],
        "predicted_matches": details["total_predicted_matches"],
        "true_matches": details["total_true_matches"],
        "blocking_recall": val_blocking_metrics["candidate_recall"],
        "blocking_pairs": val_blocking_metrics["total_candidate_pairs"],
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "xgb_params": xgb_params,
    }
    meta_path = MODELS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {meta_path.resolve()}")

    total_time = time.time() - t0
    print(f"\nPipeline finished in {total_time:.2f}s ({total_time/60:.2f} min).")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Entity Resolution Matcher")
    parser.add_argument("--samples", type=int, default=15000, help="Number of S1 entities to load for training/val")
    parser.add_argument("--val-split", type=float, default=0.20, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--no-cache", action="store_true", help="Force loading from raw TSVs instead of cache")
    args = parser.parse_args()

    train_pipeline(
        n_sample_s1=args.samples,
        val_split_ratio=args.val_split,
        random_seed=args.seed,
        use_cache=not args.no_cache,
    )
