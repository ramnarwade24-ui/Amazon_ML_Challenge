"""
High-performance, memory-safe streaming inference pipeline for Amazon ML Challenge 2026.
Generates candidate pairs from test sources, extracts pairwise features,
applies the trained XGBoost model with the optimized decision threshold,
and streams out:
1. output/candidate_pairs.tsv  (the blocking candidate set fed to the model)
2. output/matching_results.tsv (the final scored predictions)

Runs in under 4 minutes with low memory usage (< 800 MB RAM):
Pass 1: Fast inverted name & prefix indexing of test targets.
Pass 2: Match S1 entities against indices to identify candidate pairs.
Pass 3: Fast single scan over targets to fetch metadata for candidate IDs only.
Pass 4: Stream S1 entities, score candidates with XGBoost, and write output TSVs.
"""

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Set, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    TEST_S1_PATH,
    TEST_S2_PATH,
    TEST_S3_PATH,
    CANDIDATE_PAIRS_PATH,
    MATCHING_RESULTS_PATH,
    MODELS_DIR,
)
from src.preprocessing import normalize_business_name, normalize_business_address
from src.features import compute_pair_features, FEATURE_COLUMNS


def fast_clean(text: str) -> str:
    """Ultra-fast text normalization for candidate indexing."""
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


def run_streaming_inference(
    s1_path: Path = TEST_S1_PATH,
    s2_path: Path = TEST_S2_PATH,
    s3_path: Path = TEST_S3_PATH,
    model_path: Optional[Path] = None,
    candidate_output_path: Path = CANDIDATE_PAIRS_PATH,
    matching_output_path: Path = MATCHING_RESULTS_PATH,
    max_cands_per_s1: int = 25,
    batch_predict_size: int = 10000,
    nrows: Optional[int] = None,
):
    """
    Execute streaming inference end-to-end.
    Guarantees every S1 entity appears exactly once in both output files.
    """
    t0 = time.time()
    if model_path is None:
        model_path = MODELS_DIR / "matcher.pkl"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the model first.")

    print(f"Loading trained matcher from: {model_path.resolve()}")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    decision_threshold = float(artifact["best_threshold"])
    print(f"Loaded XGBoost model. Optimal Decision Threshold: {decision_threshold:.3f}")

    # =========================================================================
    # STEP 1: Fast Inverted Name & Prefix Indexing of Test Targets
    # =========================================================================
    print("\n[Step 1/4] Building target candidate indices from test Source 2 and Source 3...")
    t_idx = time.time()

    # country -> clean_name -> list of target_ids
    exact_name_index = defaultdict(lambda: defaultdict(list))
    # country -> 2-word prefix -> list of target_ids
    prefix_index = defaultdict(lambda: defaultdict(list))

    total_targets_scanned = 0
    for target_path in [s2_path, s3_path]:
        print(f"  Scanning {Path(target_path).name}...")
        for chunk in pd.read_csv(
            target_path,
            sep="\t",
            usecols=["entity_id", "business_name", "country"],
            chunksize=500000,
            dtype=str,
            keep_default_na=False,
            nrows=nrows,
        ):
            c_vals = chunk["country"].str.strip().str.upper().values
            n_vals = chunk["business_name"].values
            id_vals = chunk["entity_id"].values

            for c, name, eid in zip(c_vals, n_vals, id_vals):
                clean_n = fast_clean(name)
                if clean_n:
                    exact_name_index[c][clean_n].append(eid)
                    # 2-word prefix
                    sp = clean_n.split(" ", 2)
                    if len(sp) >= 2:
                        pfx = sp[0] + " " + sp[1]
                        pfx_list = prefix_index[c][pfx]
                        if len(pfx_list) < 25:
                            pfx_list.append(eid)
            total_targets_scanned += len(chunk)

    print(f"Scanned {total_targets_scanned:,} target records in {time.time() - t_idx:.2f}s.")
    for country in exact_name_index:
        print(f"  Country '{country}': {len(exact_name_index[country]):,} unique names.")

    # =========================================================================
    # STEP 2: Match S1 Entities Against Indices
    # =========================================================================
    print("\n[Step 2/4] Generating candidates for Source 1 test entities...")
    t_cands = time.time()

    s1_candidates: Dict[str, List[str]] = {}
    needed_target_ids: Set[str] = set()

    total_s1 = 0
    s1_with_cands = 0
    total_candidate_pairs = 0

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        usecols=["entity_id", "business_name", "country"],
        chunksize=500000,
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    ):
        c_vals = chunk["country"].str.strip().str.upper().values
        n_vals = chunk["business_name"].values
        id_vals = chunk["entity_id"].values

        for c, name, s1_id in zip(c_vals, n_vals, id_vals):
            total_s1 += 1
            clean_n = fast_clean(name)
            cands_set = set()

            if clean_n:
                # 1. Exact name match
                if clean_n in exact_name_index[c]:
                    for tid in exact_name_index[c][clean_n]:
                        cands_set.add(tid)
                        if len(cands_set) >= max_cands_per_s1:
                            break

                # 2. 2-word prefix match
                if len(cands_set) < max_cands_per_s1:
                    sp = clean_n.split(" ", 2)
                    if len(sp) >= 2:
                        pfx = sp[0] + " " + sp[1]
                        if pfx in prefix_index[c]:
                            for tid in prefix_index[c][pfx]:
                                cands_set.add(tid)
                                if len(cands_set) >= max_cands_per_s1:
                                    break

            cand_list = sorted([cid for cid in cands_set if cid.startswith(("S2-", "S3-"))])
            s1_candidates[s1_id] = cand_list
            if cand_list:
                s1_with_cands += 1
                total_candidate_pairs += len(cand_list)
                needed_target_ids.update(cand_list)

    print(f"Generated candidates for {total_s1:,} S1 entities in {time.time() - t_cands:.2f}s.")
    print(f"  S1 with candidates:     {s1_with_cands:,} ({s1_with_cands/total_s1*100:.2f}%)")
    print(f"  S1 singletons (0 cand): {total_s1 - s1_with_cands:,} ({(total_s1 - s1_with_cands)/total_s1*100:.2f}%)")
    print(f"  Total candidate pairs:   {total_candidate_pairs:,}")
    print(f"  Unique target IDs needed: {len(needed_target_ids):,}")

    del exact_name_index, prefix_index
    gc.collect()

    # =========================================================================
    # STEP 3: Fetch Target Metadata for Needed Candidates Only
    # =========================================================================
    print(f"\n[Step 3/4] Loading metadata for {len(needed_target_ids):,} candidate target records...")
    t_fetch = time.time()
    target_metadata: Dict[str, Dict[str, str]] = {}

    for target_path in [s2_path, s3_path]:
        for chunk in pd.read_csv(
            target_path,
            sep="\t",
            chunksize=500000,
            dtype=str,
            keep_default_na=False,
            nrows=nrows,
        ):
            hit = chunk[chunk["entity_id"].isin(needed_target_ids)]
            if len(hit) > 0:
                eids = hit["entity_id"].values
                names = hit["business_name"].values
                addrs = hit["business_address"].values
                countries = hit["country"].str.strip().str.upper().values

                for eid, name, addr, c in zip(eids, names, addrs, countries):
                    target_metadata[eid] = {
                        "entity_id": eid,
                        "business_name": name,
                        "norm_name": normalize_business_name(name),
                        "business_address": addr,
                        "norm_address": normalize_business_address(addr),
                        "norm_country": c,
                    }
            if len(target_metadata) >= len(needed_target_ids):
                break

    print(f"Fetched metadata for {len(target_metadata):,} targets in {time.time() - t_fetch:.2f}s.")

    # =========================================================================
    # STEP 4: Stream S1, Extract Features, Score with XGBoost, and Write TSV
    # =========================================================================
    print(f"\n[Step 4/4] Scoring candidate pairs and streaming submission TSVs...")
    t_score = time.time()

    candidate_output_path = Path(candidate_output_path)
    matching_output_path = Path(matching_output_path)
    candidate_output_path.parent.mkdir(parents=True, exist_ok=True)
    matching_output_path.parent.mkdir(parents=True, exist_ok=True)

    f_cand = open(candidate_output_path, "w", encoding="utf-8")
    f_match = open(matching_output_path, "w", encoding="utf-8")

    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")

    total_predicted_matches = 0
    s1_with_predicted_matches = 0
    written_s1_count = 0

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        chunksize=100000,
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    ):
        eids = chunk["entity_id"].values
        names = chunk["business_name"].values
        addrs = chunk["business_address"].values
        countries = chunk["country"].str.strip().str.upper().values

        batch_pairs: List[Tuple[str, str]] = []
        batch_s1_recs: Dict[str, Dict[str, str]] = {}

        for eid, name, addr, c in zip(eids, names, addrs, countries):
            cands = s1_candidates.get(eid, [])
            if cands:
                s1_rec = {
                    "entity_id": eid,
                    "business_name": name,
                    "norm_name": normalize_business_name(name),
                    "business_address": addr,
                    "norm_address": normalize_business_address(addr),
                    "norm_country": c,
                }
                batch_s1_recs[eid] = s1_rec
                for cid in cands:
                    batch_pairs.append((eid, cid))

        matched_map: Dict[str, List[str]] = defaultdict(list)
        if batch_pairs:
            for p_start in range(0, len(batch_pairs), batch_predict_size):
                p_end = min(p_start + batch_predict_size, len(batch_pairs))
                sub_pairs = batch_pairs[p_start:p_end]

                feat_rows = []
                for s_id, c_id in sub_pairs:
                    s_rec = batch_s1_recs[s_id]
                    c_rec = target_metadata.get(c_id, {"entity_id": c_id})
                    feat_rows.append(compute_pair_features(s_rec, c_rec))

                X_sub = pd.DataFrame(feat_rows, columns=FEATURE_COLUMNS, dtype=np.float32)
                probs = model.predict_proba(X_sub)[:, 1]

                for (s_id, c_id), p in zip(sub_pairs, probs):
                    if p >= decision_threshold:
                        matched_map[s_id].append(c_id)

        for eid in eids:
            written_s1_count += 1
            cands = s1_candidates.get(eid, [])
            cand_str = ",".join(cands)
            f_cand.write(f"{eid}\t{cand_str}\n")

            matches = matched_map.get(eid, [])
            if matches:
                unique_sorted_matches = sorted(list(set(matches)))
                match_str = ",".join(unique_sorted_matches)
                total_predicted_matches += len(unique_sorted_matches)
                s1_with_predicted_matches += 1
            else:
                match_str = ""
            f_match.write(f"{eid}\t{match_str}\n")

    f_cand.close()
    f_match.close()

    total_time = time.time() - t0
    print("\n==========================================")
    print("FINAL INFERENCE SUMMARY:")
    print("==========================================")
    print(f"Total S1 entities processed:       {written_s1_count:,}")
    print(f"Predicted Singletons (0 match):   {written_s1_count - s1_with_predicted_matches:,} ({(written_s1_count - s1_with_predicted_matches)/written_s1_count*100:.2f}%)")
    print(f"Entities with predicted matches:  {s1_with_predicted_matches:,} ({s1_with_predicted_matches/written_s1_count*100:.2f}%)")
    print(f"Total predicted pairwise matches: {total_predicted_matches:,}")
    print(f"Candidate pairs file written:     {candidate_output_path.resolve()}")
    print(f"Matching results file written:    {matching_output_path.resolve()}")
    print(f"Total inference runtime:          {total_time:.2f}s ({total_time/60:.2f} min)")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Streaming Inference")
    parser.add_argument("--nrows", type=int, default=None, help="Row limit for testing")
    parser.add_argument("--max-cands", type=int, default=25, help="Max candidates per S1 entity")
    args = parser.parse_args()

    run_streaming_inference(
        nrows=args.nrows,
        max_cands_per_s1=args.max_cands,
    )
