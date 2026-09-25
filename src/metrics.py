"""
Evaluation metrics for Amazon ML Challenge 2026.
Implements the exact macro-averaged F_0.5 score as defined in the official guidelines.
Also includes blocking quality metrics (Candidate Recall and Reduction Ratio).
"""

from typing import Dict, Set, Iterable, Tuple, List
import numpy as np

def compute_entity_f_beta(
    true_matches: Set[str],
    pred_matches: Set[str],
    beta: float = 0.5,
) -> float:
    """
    Compute F_beta for a single Source 1 entity.
    
    Rules according to challenge specification:
    - If true is empty (singleton) and predicted is empty: F_beta = 1.0
    - If true is empty (singleton) and predicted is non-empty: F_beta = 0.0
    - If true is non-empty and predicted is empty: F_beta = 0.0
    - If precision + recall == 0: F_beta = 0.0
    """
    len_true = len(true_matches)
    len_pred = len(pred_matches)
    
    if len_true == 0:
        return 1.0 if len_pred == 0 else 0.0
    
    if len_pred == 0:
        return 0.0
    
    tp = len(true_matches.intersection(pred_matches))
    if tp == 0:
        return 0.0
    
    precision = tp / len_pred
    recall = tp / len_true
    
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0
    
    return ((1.0 + beta_sq) * precision * recall) / denom


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate macro-averaged F_0.5 score across all Source 1 entities in ground_truth.
    
    Parameters
    ----------
    ground_truth : Dict[str, Set[str]]
        Mapping of source1_entity_id -> set of true matched entity IDs.
    predictions : Dict[str, Set[str]]
        Mapping of source1_entity_id -> set of predicted matched entity IDs.
        
    Returns
    -------
    macro_score : float
        Mean F_beta score across all S1 entities.
    details : dict
        Breakdown containing singleton accuracy, non-singleton F_beta, etc.
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f_beta(true_set, pred_set, beta=beta)
        scores.append(score)
        
        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)
            
    macro_score = float(np.mean(scores)) if scores else 0.0
    
    details = {
        "macro_f05": macro_score,
        "num_entities": len(ground_truth),
        "singleton_f05": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
        "num_singletons": len(singleton_scores),
        "num_non_singletons": len(non_singleton_scores),
    }
    return macro_score, details


def evaluate_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
) -> Dict[str, float]:
    """
    Evaluate the quality of candidate generation (blocking):
    - Candidate Recall: proportion of true links captured in candidate sets
    - Entity Recall: proportion of non-singleton S1 entities where at least 1 match was found
    - Complete Match Recall: proportion of S1 entities where ALL true matches were found
    - Average Candidates per S1 entity
    """
    total_true_links = 0
    captured_links = 0
    
    entities_with_matches = 0
    entities_partially_captured = 0
    entities_fully_captured = 0
    
    total_candidates = 0
    
    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        total_candidates += len(cand_set)
        
        n_true = len(true_set)
        if n_true == 0:
            continue
            
        total_true_links += n_true
        entities_with_matches += 1
        
        intersection = true_set.intersection(cand_set)
        captured_links += len(intersection)
        
        if len(intersection) > 0:
            entities_partially_captured += 1
        if len(intersection) == n_true:
            entities_fully_captured += 1
            
    candidate_recall = captured_links / total_true_links if total_true_links > 0 else 0.0
    entity_recall = entities_partially_captured / entities_with_matches if entities_with_matches > 0 else 0.0
    complete_recall = entities_fully_captured / entities_with_matches if entities_with_matches > 0 else 0.0
    avg_candidates = total_candidates / len(ground_truth) if ground_truth else 0.0
    
    return {
        "candidate_recall": candidate_recall,
        "entity_partial_recall": entity_recall,
        "entity_complete_recall": complete_recall,
        "total_true_links": total_true_links,
        "captured_true_links": captured_links,
        "total_candidate_pairs": total_candidates,
        "avg_candidates_per_s1": avg_candidates,
    }
