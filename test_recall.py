import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter

cache_dir = Path("Amazon_ML_Challenge/models/benchmark_cache")
s1_df = pd.read_csv(cache_dir / "sample_s1.tsv", sep="\t", keep_default_na=False)
gt_df = pd.read_csv(cache_dir / "sample_gt.tsv", sep="\t", keep_default_na=False)
targets_df = pd.read_csv(cache_dir / "sample_targets.tsv", sep="\t", keep_default_na=False)
gt_map = {row["source1_entity_id"]: {x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()} for _, row in gt_df.iterrows()}

sys.path.insert(0, "Amazon_ML_Challenge/code/business_entity_resolution")
from src.preprocessing import preprocess_dataframe
from src.metrics import evaluate_blocking_recall
from src.blocking import MultiPassBlocker

s1_norm = preprocess_dataframe(s1_df)
s2_norm = preprocess_dataframe(targets_df[targets_df["entity_id"].str.startswith("S2-")])
s3_norm = preprocess_dataframe(targets_df[targets_df["entity_id"].str.startswith("S3-")])

b = MultiPassBlocker()
target_df = pd.concat([s2_norm, s3_norm], ignore_index=True)

all_countries = sorted(list(s1_norm["norm_country"].unique()))
final_candidates = {s1_id: set() for s1_id in s1_norm["entity_id"].values}

for country in all_countries:
    s1_sub = s1_norm[s1_norm["norm_country"] == country]
    target_sub = target_df[target_df["norm_country"] == country]
    if len(s1_sub) == 0 or len(target_sub) == 0:
        continue
    
    s1_ids = s1_sub["entity_id"].values
    s1_names = s1_sub["norm_name"].values
    s1_addrs = s1_sub["norm_address"].values
    target_ids = target_sub["entity_id"].values
    target_names = target_sub["norm_name"].values
    target_addrs = target_sub["norm_address"].values

    p1 = b.pass1_exact_name(s1_names, s1_ids, target_names, target_ids)
    p2 = b.pass2_name_tokens(s1_names, s1_ids, target_names, target_ids)
    p3 = b.pass3_address_tokens(s1_addrs, s1_ids, target_addrs, target_ids)
    p4 = b.pass4_tfidf_names(s1_names, s1_ids, target_names, target_ids)
    p5 = b.pass5_tfidf_addrs(s1_addrs, s1_ids, target_addrs, target_ids)

    for s1_id in s1_ids:
        cand_scores = Counter()
        for cid in p1.get(s1_id, set()):
            cand_scores[cid] += 10.0
        for cid in p4.get(s1_id, set()):
            cand_scores[cid] += 6.0
        for cid in p5.get(s1_id, set()):
            cand_scores[cid] += 5.0
        for cid in p3.get(s1_id, set()):
            cand_scores[cid] += 4.0
        for cid in p2.get(s1_id, set()):
            cand_scores[cid] += 1.0

        # Select top-35 by priority score
        ranked = [cid for cid, score in cand_scores.most_common(35) if cid.startswith(("S2-", "S3-"))]
        final_candidates[s1_id] = set(ranked)

m = evaluate_blocking_recall(gt_map, final_candidates)
rec = m["candidate_recall"] * 100
print(f"Priority-Ranked Blocking Recall (cap=35): {rec:.2f}% (captured: {m['captured_true_links']}/{m['total_true_links']})")
print(f"Entity Partial Recall: {m['entity_partial_recall']*100:.2f}%")
print(f"Entity Complete Recall: {m['entity_complete_recall']*100:.2f}%")
print(f"Total candidate pairs: {m['total_candidate_pairs']:,}")
print(f"Avg candidates per S1: {m['avg_candidates_per_s1']:.2f}")
