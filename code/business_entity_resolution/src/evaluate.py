"""
Evaluation module and CLI for Amazon ML Challenge 2026.
Evaluates the complete entity resolution pipeline (blocking + XGBoost matcher)
on held-out validation data using the official macro F_0.5 metric, singleton metrics,
and blocking recall ceiling.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Set, Optional

import joblib
import numpy as np
import pandas as pd

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TRAIN_S1_PATH,
    TRAIN_S2_PATH,
    TRAIN_S3_PATH,
    TRAIN_GT_PATH,
    MODELS_DIR,
    F_BETA,
    RANDOM_SEED,
)
from src.data_loader import load_source_file, load_ground_truth
from src.preprocessing import preprocess_dataframe
from src.blocking import MultiPassBlocker
from src.features import extract_features_for_pairs, FEATURE_COLUMNS
from src.metrics import compute_macro_f05, evaluate_blocking_recall


def evaluate_matcher(
    model_artifact_path: Optional[Path] = None,
    n_sample_s1: int = 5000,
    threshold_override: Optional[float] = None,
    random_seed: int = RANDOM_SEED,
):
    """
    Run evaluation using the saved model artifact against held-out validation S1 entities.
    """
    if model_artifact_path is None:
        model_artifact_path = MODELS_DIR / "matcher.pkl"

    if not model_artifact_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_artifact_path}. Please run train_model.py first."
        )

    print(f"Loading trained model artifact from: {model_artifact_path.resolve()}")
    artifact = joblib.load(model_artifact_path)
    model = artifact["model"]
    decision_threshold = threshold_override if threshold_override is not None else artifact["best_threshold"]

    print(f"Using Decision Threshold: {decision_threshold:.3f}")

    print(f"\nLoading validation sample of {n_sample_s1:,} S1 records from training ground truth...")
    df_gt, gt_map = load_ground_truth(TRAIN_GT_PATH, nrows=n_sample_s1)
    s1_ids = list(df_gt["source1_entity_id"].values)

    all_s1_set = set(s1_ids)
    s1_list = []
    for chunk in pd.read_csv(TRAIN_S1_PATH, sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
        hit = chunk[chunk["entity_id"].isin(all_s1_set)]
        if len(hit) > 0:
            s1_list.append(hit)
        if sum(len(x) for x in s1_list) >= len(s1_ids):
            break
    s1_df = pd.concat(s1_list, ignore_index=True).drop_duplicates(subset=["entity_id"])

    # Collect true targets
    target_ids = set()
    for sid in s1_ids:
        target_ids.update(gt_map.get(sid, set()))

    s2_list = []
    s3_list = []
    for chunk in pd.read_csv(TRAIN_S2_PATH, sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
        hit = chunk[chunk["entity_id"].isin(target_ids)]
        if len(hit) > 0:
            s2_list.append(hit)
        if len(s2_list) == 1:
            s2_list.append(chunk.head(5000))
        if sum(len(x) for x in s2_list) >= len(target_ids) // 2 + 10000:
            break

    for chunk in pd.read_csv(TRAIN_S3_PATH, sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
        hit = chunk[chunk["entity_id"].isin(target_ids)]
        if len(hit) > 0:
            s3_list.append(hit)
        if len(s3_list) == 1:
            s3_list.append(chunk.head(5000))
        if sum(len(x) for x in s3_list) >= len(target_ids) // 2 + 10000:
            break

    s2_df = pd.concat(s2_list, ignore_index=True).drop_duplicates(subset=["entity_id"])
    s3_df = pd.concat(s3_list, ignore_index=True).drop_duplicates(subset=["entity_id"])

    # Normalization
    s1_norm = preprocess_dataframe(s1_df)
    s2_norm = preprocess_dataframe(s2_df)
    s3_norm = preprocess_dataframe(s3_df)

    s1_dict = s1_norm.set_index("entity_id").to_dict("index")
    target_norm = pd.concat([s2_norm, s3_norm], ignore_index=True).drop_duplicates(subset=["entity_id"])
    target_dict = target_norm.set_index("entity_id").to_dict("index")

    # Multi-pass blocking
    blocker = MultiPassBlocker()
    candidates, blocking_stats = blocker.generate_candidates(s1_norm, s2_norm, s3_norm, verbose=False)

    # Evaluate blocking recall ceiling
    pool_ids = set(target_norm["entity_id"])
    gt_eval = {sid: {m for m in gt_map.get(sid, set()) if m in pool_ids} for sid in s1_ids}
    blocking_metrics = evaluate_blocking_recall(gt_eval, candidates)

    # Prepare candidate pairs for classification
    eval_pairs = []
    for s1_id in s1_ids:
        cands = candidates.get(s1_id, set())
        for cand_id in cands:
            eval_pairs.append((s1_id, cand_id))

    print(f"Candidate pairs to score: {len(eval_pairs):,}")
    if eval_pairs:
        X_eval = extract_features_for_pairs(eval_pairs, s1_dict, target_dict)
        probs = model.predict_proba(X_eval)[:, 1]
    else:
        probs = np.array([])

    # Group predictions by s1_id
    preds = {s1_id: set() for s1_id in s1_ids}
    for (s1_id, cand_id), p in zip(eval_pairs, probs):
        if p >= decision_threshold:
            preds[s1_id].add(cand_id)

    # Compute Macro F_0.5 and breakdown
    macro_score, details = compute_macro_f05(gt_eval, preds, beta=F_BETA)

    # Compute overall precision and recall
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for s1_id in s1_ids:
        pred_set = preds[s1_id]
        true_set = gt_eval.get(s1_id, set())
        tp = len(pred_set.intersection(true_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    print("\n==========================================")
    print("      ENTITY RESOLUTION EVALUATION REPORT ")
    print("==========================================")
    print(f"Validation S1 Entities:     {len(s1_ids):,}")
    print(f"Decision Threshold:         {decision_threshold:.3f}")
    print(f"--- Blocking Quality ---")
    print(f"Candidate Recall Ceiling:   {blocking_metrics['candidate_recall']*100:.2f}%")
    print(f"Entity Partial Recall:      {blocking_metrics['entity_partial_recall']*100:.2f}%")
    print(f"Entity Complete Recall:     {blocking_metrics['entity_complete_recall']*100:.2f}%")
    print(f"Missed Matches in Blocking: {blocking_metrics['total_true_links'] - blocking_metrics['captured_true_links']:,}")
    print(f"Total Candidate Pairs:      {blocking_metrics['total_candidate_pairs']:,}")
    print(f"Avg Candidates / S1:        {blocking_metrics['avg_candidates_per_s1']:.2f}")
    print(f"\n--- Classification & Matching ---")
    print(f"Macro F_0.5 Score:          {macro_score:.4f} (Primary Challenge Metric)")
    print(f"Overall Precision:          {precision*100:.2f}%")
    print(f"Overall Recall:             {recall*100:.2f}%")
    print(f"Total Predicted Matches:    {total_tp + total_fp:,}")
    print(f"Total True Matches:         {total_tp + total_fn:,}")
    print(f"True Positives:             {total_tp:,}")
    print(f"False Positives:            {total_fp:,}")
    print(f"False Negatives:            {total_fn:,}")
    print(f"\n--- Singleton Analysis ---")
    print(f"Singleton Count:            {details['num_singletons']:,}")
    print(f"Singleton Accuracy (F0.5):  {details['singleton_f05']*100:.2f}%")
    print(f"Non-Singleton Count:        {details['num_non_singletons']:,}")
    print(f"Non-Singleton F0.5:         {details['non_singleton_f05']:.4f}")
    print("==========================================")

    return {
        "macro_f05": macro_score,
        "precision": precision,
        "recall": recall,
        "singleton_f05": details["singleton_f05"],
        "non_singleton_f05": details["non_singleton_f05"],
        "blocking_recall": blocking_metrics["candidate_recall"],
        "total_predicted_matches": total_tp + total_fp,
        "total_true_matches": total_tp + total_fn,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution Matching Pipeline")
    parser.add_argument("--samples", type=int, default=5000, help="Number of S1 validation entities")
    parser.add_argument("--threshold", type=float, default=None, help="Probability threshold override")
    args = parser.parse_args()

    evaluate_matcher(
        n_sample_s1=args.samples,
        threshold_override=args.threshold,
    )
