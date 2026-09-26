"""
High-performance, memory-safe inverted streaming inference pipeline for Amazon ML Challenge 2026.
Indexes test Source 1 entities in memory (< 400 MB RAM), streams test Source 2 and Source 3 targets,
generates high-recall multi-evidence candidates, fetches target metadata, scores candidates with XGBoost,
and streams out:
1. output/candidate_pairs.tsv  (the blocking candidate set fed to the model)
2. output/matching_results.tsv (the final scored predictions)

Guarantees:
- Every Source 1 test entity appears exactly once in both files.
- Strictly open-set (dynamically processes US, India, France, and any test country).
- Final matches are a strict subset of candidate pairs.
- Multi-match support (zero, one, or multiple matches per S1 entity).
- Memory-safe (< 1.5 GB RAM) and fast execution.
"""

import argparse
import gc
import re
import sys
import time
from collections import defaultdict, Counter
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

DOMAIN_PATTERN = re.compile(r"\.(?:com|org|net|in|fr|co\.in|co|io)\b", re.I)
NUM_PATTERN = re.compile(r"\b\d{2,}\b")

GENERIC_NAME_STOPWORDS = {
    "and", "the", "of", "in", "at", "for", "to", "a", "an",
    "services", "solutions", "enterprises", "center", "centre",
    "group", "associates", "consultants", "holdings", "systems",
    "technologies", "international", "national", "global", "industries",
}

GENERIC_ADDR_STOPWORDS = {
    "near", "opp", "opposite", "road", "street", "st", "rd", "ave", "avenue",
    "lane", "ln", "dr", "drive", "blvd", "boulevard", "floor", "fl", "suite",
    "ste", "apt", "apartment", "bldg", "building", "null", "none", "nan",
    "city", "town", "post", "box", "pob", "cross", "main", "sector", "block",
}


def run_streaming_inference(
    s1_path: Path = TEST_S1_PATH,
    s2_path: Path = TEST_S2_PATH,
    s3_path: Path = TEST_S3_PATH,
    model_path: Optional[Path] = None,
    candidate_output_path: Path = CANDIDATE_PAIRS_PATH,
    matching_output_path: Path = MATCHING_RESULTS_PATH,
    max_cands_per_s1: int = 35,
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
    # STEP 1: Inverted Indexing of Test Source 1 Entities in Memory
    # =========================================================================
    print("\n[Step 1/4] Building S1 candidate indices (memory-efficient inverted architecture)...")
    t_idx = time.time()

    s1_exact = defaultdict(lambda: defaultdict(list))
    s1_pfx = defaultdict(lambda: defaultdict(list))
    s1_name_tok = defaultdict(lambda: defaultdict(list))
    s1_addr_tok = defaultdict(lambda: defaultdict(list))
    s1_num = defaultdict(lambda: defaultdict(list))
    s1_global_exact = defaultdict(list)

    all_s1_ids_order: List[str] = []

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        chunksize=250000,
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    ):
        eids = chunk["entity_id"].values
        names = chunk["business_name"].values
        addrs = chunk["business_address"].values
        countries = chunk["country"].str.strip().str.upper().values

        for eid, name, addr, c in zip(eids, names, addrs, countries):
            all_s1_ids_order.append(eid)

            n_norm = normalize_business_name(name)
            a_norm = normalize_business_address(addr)

            if n_norm:
                s1_exact[c][n_norm].append(eid)
                s1_global_exact[n_norm].append(eid)

                stripped = DOMAIN_PATTERN.sub("", n_norm).strip()
                if stripped and stripped != n_norm:
                    s1_exact[c][stripped].append(eid)
                    s1_global_exact[stripped].append(eid)

                words = n_norm.split()
                if len(words) >= 2:
                    pfx = words[0] + " " + words[1]
                    s1_pfx[c][pfx].append(eid)

                for tok in set(words):
                    if len(tok) >= 3 and tok not in GENERIC_NAME_STOPWORDS:
                        s1_name_tok[c][tok].append(eid)

            if a_norm:
                a_words = a_norm.split()
                for tok in set(a_words):
                    if len(tok) >= 4 and tok not in GENERIC_ADDR_STOPWORDS and not tok.isdigit():
                        s1_addr_tok[c][tok].append(eid)

                for num in set(NUM_PATTERN.findall(a_norm)):
                    s1_num[c][num].append(eid)

    total_s1 = len(all_s1_ids_order)
    print(f"Indexed {total_s1:,} Source 1 test records across countries: {list(s1_exact.keys())} in {time.time() - t_idx:.2f}s.", flush=True)

    # Prune high-frequency tokens to maintain strict memory safety and avoid candidate pollution
    max_token_freq = 50
    print(f"Pruning token indices with frequency > {max_token_freq} for memory safety...", flush=True)
    for c in list(s1_name_tok.keys()):
        s1_name_tok[c] = {k: v for k, v in s1_name_tok[c].items() if len(v) <= max_token_freq}
        s1_pfx[c] = {k: v for k, v in s1_pfx[c].items() if len(v) <= max_token_freq}
        s1_addr_tok[c] = {k: v for k, v in s1_addr_tok[c].items() if len(v) <= max_token_freq}
        s1_num[c] = {k: v for k, v in s1_num[c].items() if len(v) <= max_token_freq}

    # =========================================================================
    # STEP 2: Stream Test Targets (S2 + S3) and Accumulate Multi-Pass Evidence
    # =========================================================================
    print("\n[Step 2/4] Streaming target records from test Source 2 and Source 3...", flush=True)
    t_stream = time.time()
    total_targets_scanned = 0

    s1_candidates: Dict[str, Counter] = defaultdict(Counter)

    def add_cand(sid: str, tid: str, score: float):
        c = s1_candidates[sid]
        c[tid] += score
        if len(c) > 60:
            s1_candidates[sid] = Counter(dict(c.most_common(35)))

    for target_path in [s2_path, s3_path]:
        p = Path(target_path)
        if not p.is_file():
            print(f"  Warning: {p} not found, skipping.", flush=True)
            continue
        print(f"  Scanning {p.name}...", flush=True)
        for chunk in pd.read_csv(
            target_path,
            sep="\t",
            chunksize=500000,
            dtype=str,
            keep_default_na=False,
            nrows=nrows,
        ):
            c_vals = chunk["country"].str.strip().str.upper().values
            n_vals = chunk["business_name"].values
            a_vals = chunk["business_address"].values
            id_vals = chunk["entity_id"].values

            for c, name, addr, tid in zip(c_vals, n_vals, a_vals, id_vals):
                n_norm = normalize_business_name(name)
                a_norm = normalize_business_address(addr)

                if n_norm:
                    # Pass 1: Exact name
                    if n_norm in s1_exact[c]:
                        for sid in s1_exact[c][n_norm]:
                            add_cand(sid, tid, 10.0)
                    stripped = DOMAIN_PATTERN.sub("", n_norm).strip()
                    if stripped and stripped in s1_exact[c]:
                        for sid in s1_exact[c][stripped]:
                            add_cand(sid, tid, 10.0)

                    # Pass 2: 2-word prefix
                    words = n_norm.split()
                    if len(words) >= 2:
                        pfx = words[0] + " " + words[1]
                        if pfx in s1_pfx[c]:
                            for sid in s1_pfx[c][pfx]:
                                add_cand(sid, tid, 5.0)

                    # Pass 2: Name tokens
                    for tok in set(words):
                        if tok in s1_name_tok[c]:
                            for sid in s1_name_tok[c][tok]:
                                add_cand(sid, tid, 2.0)

                if a_norm:
                    # Pass 4: Address tokens
                    a_words = a_norm.split()
                    for tok in set(a_words):
                        if tok in s1_addr_tok[c]:
                            for sid in s1_addr_tok[c][tok]:
                                add_cand(sid, tid, 3.0)

                    for num in set(NUM_PATTERN.findall(a_norm)):
                        if num in s1_num[c]:
                            for sid in s1_num[c][num]:
                                add_cand(sid, tid, 3.0)

            total_targets_scanned += len(chunk)
            print(f"    Scanned {total_targets_scanned:,} targets so far ({time.time() - t_stream:.1f}s)...", flush=True)

    print(f"Scanned {total_targets_scanned:,} target records in {time.time() - t_stream:.2f}s.", flush=True)

    # Rank top candidates per S1 entity and collect needed target IDs
    print("\nRanking top candidate pairs per S1 entity...", flush=True)
    final_s1_candidates: Dict[str, List[str]] = {}
    needed_target_ids: Set[str] = set()
    s1_with_cands = 0
    total_candidate_pairs = 0

    for sid in all_s1_ids_order:
        scores = s1_candidates[sid]
        ranked = [tid for tid, score in scores.most_common(max_cands_per_s1) if tid.startswith(("S2-", "S3-"))]
        final_s1_candidates[sid] = ranked
        if ranked:
            s1_with_cands += 1
            total_candidate_pairs += len(ranked)
            needed_target_ids.update(ranked)

    print(f"Candidate generation summary:", flush=True)
    print(f"  Total S1 entities:       {total_s1:,}", flush=True)
    print(f"  S1 with candidates:     {s1_with_cands:,} ({s1_with_cands/(total_s1+1e-9)*100:.2f}%)", flush=True)
    print(f"  S1 singletons (0 cand): {total_s1 - s1_with_cands:,} ({(total_s1 - s1_with_cands)/(total_s1+1e-9)*100:.2f}%)", flush=True)
    print(f"  Total candidate pairs:   {total_candidate_pairs:,}", flush=True)
    print(f"  Average candidates/S1:  {total_candidate_pairs/(total_s1+1e-9):.2f}", flush=True)
    print(f"  Unique target IDs needed: {len(needed_target_ids):,}", flush=True)

    del s1_exact, s1_pfx, s1_name_tok, s1_addr_tok, s1_num, s1_global_exact, s1_candidates
    gc.collect()

    # =========================================================================
    # STEP 3: Fetch Target Metadata for Needed Candidates Only
    # =========================================================================
    print(f"\n[Step 3/4] Loading metadata for {len(needed_target_ids):,} candidate target records...")
    t_fetch = time.time()
    target_metadata: Dict[str, Dict[str, str]] = {}

    for target_path in [s2_path, s3_path]:
        p = Path(target_path)
        if not p.is_file():
            continue
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
    # STEP 4: Stream S1, Extract Features, Score with XGBoost, and Write TSVs
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
            cands = final_s1_candidates.get(eid, [])
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
            cands = final_s1_candidates.get(eid, [])
            cand_str = ",".join(cands)
            f_cand.write(f"{eid}\t{cand_str}\n")

            matches = matched_map.get(eid, [])
            if matches:
                # Deduplicate and sort matches
                unique_sorted_matches = sorted(list(set(matches)))
                match_str = ",".join(unique_sorted_matches)
                total_predicted_matches += len(unique_sorted_matches)
                s1_with_predicted_matches += 1
            else:
                match_str = ""
            f_match.write(f"{eid}\t{match_str}\n")

        print(f"    Processed {written_s1_count:,}/{total_s1:,} S1 records ({time.time() - t_score:.1f}s)...", flush=True)

    f_cand.close()
    f_match.close()

    total_time = time.time() - t0
    print("\n==========================================")
    print("FINAL INFERENCE SUMMARY:")
    print("==========================================")
    print(f"Total S1 entities processed:       {written_s1_count:,}")
    print(f"Predicted Singletons (0 match):   {written_s1_count - s1_with_predicted_matches:,} ({(written_s1_count - s1_with_predicted_matches)/(written_s1_count+1e-9)*100:.2f}%)")
    print(f"Entities with predicted matches:  {s1_with_predicted_matches:,} ({s1_with_predicted_matches/(written_s1_count+1e-9)*100:.2f}%)")
    print(f"Total predicted pairwise matches: {total_predicted_matches:,}")
    print(f"Candidate pairs file written:     {candidate_output_path.resolve()}")
    print(f"Matching results file written:    {matching_output_path.resolve()}")
    print(f"Total inference runtime:          {total_time:.2f}s ({total_time/60:.2f} min)")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Streaming Inference")
    parser.add_argument("--nrows", type=int, default=None, help="Row limit for testing")
    parser.add_argument("--max-cands", type=int, default=35, help="Max candidates per S1 entity")
    args = parser.parse_args()

    run_streaming_inference(
        nrows=args.nrows,
        max_cands_per_s1=args.max_cands,
    )
