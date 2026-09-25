"""
Multi-pass candidate generation (blocking) module for Amazon ML Challenge 2026.
Generates candidates from Source 1 to Source 2 and Source 3 using 5 complementary passes:
1. Exact normalized name blocking
2. Normalized name token/prefix blocking
3. Address-token and number blocking
4. Character n-gram TF-IDF similarity retrieval for names
5. Character n-gram TF-IDF similarity retrieval for addresses

Strictly open-set (groups dynamically by country label without hardcoding).
Uses sparse TF-IDF matrices and vectorized matrix multiplications for speed and efficiency.
"""

from collections import defaultdict
import gc
import re
from typing import Dict, Set, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import BLOCKING_CONFIG


class MultiPassBlocker:
    """
    Multi-pass candidate generator operating across data sources.
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or BLOCKING_CONFIG
        self.min_name_token_len = self.cfg.get("min_name_token_len", 3)
        self.min_addr_token_len = self.cfg.get("min_addr_token_len", 4)
        self.max_token_freq = self.cfg.get("max_token_frequency", 5000)
        self.tfidf_name_k = self.cfg.get("tfidf_name_top_k", 5)
        self.tfidf_name_thresh = self.cfg.get("tfidf_name_threshold", 0.45)
        self.tfidf_addr_k = self.cfg.get("tfidf_addr_top_k", 5)
        self.tfidf_addr_thresh = self.cfg.get("tfidf_addr_threshold", 0.40)
        self.max_candidates = self.cfg.get("max_candidates_per_s1", 50)

    # -------------------------------------------------------------
    # PASS 1: Exact Normalized Name Blocking
    # -------------------------------------------------------------
    def pass1_exact_name(
        self, s1_names: np.ndarray, s1_ids: np.ndarray, target_names: np.ndarray, target_ids: np.ndarray
    ) -> Dict[str, Set[str]]:
        """Match exact normalized business names."""
        name_index = defaultdict(list)
        for name, tid in zip(target_names, target_ids):
            if name:
                name_index[name].append(tid)

        candidates = defaultdict(set)
        for s1_id, name in zip(s1_ids, s1_names):
            if name and name in name_index:
                candidates[s1_id].update(name_index[name])

        return candidates

    # -------------------------------------------------------------
    # PASS 2: Normalized Name Token & Prefix Blocking
    # -------------------------------------------------------------
    def pass2_name_tokens(
        self, s1_names: np.ndarray, s1_ids: np.ndarray, target_names: np.ndarray, target_ids: np.ndarray
    ) -> Dict[str, Set[str]]:
        """Match significant name tokens and distinctive prefixes."""
        token_index = defaultdict(list)
        for name, tid in zip(target_names, target_ids):
            if not name:
                continue
            tokens = [t for t in name.split() if len(t) >= self.min_name_token_len]
            # Use unique tokens per record
            for token in set(tokens):
                token_index[token].append(tid)

        # Filter out hyper-frequent stopwords/tokens that would explode candidates
        token_index = {
            t: ids for t, ids in token_index.items() if len(ids) <= self.max_token_freq
        }

        candidates = defaultdict(set)
        for s1_id, name in zip(s1_ids, s1_names):
            if not name:
                continue
            tokens = [t for t in name.split() if len(t) >= self.min_name_token_len]
            for token in set(tokens):
                if token in token_index:
                    candidates[s1_id].update(token_index[token])

        return candidates

    # -------------------------------------------------------------
    # PASS 3: Address Token & Number Blocking
    # -------------------------------------------------------------
    def pass3_address_tokens(
        self, s1_addrs: np.ndarray, s1_ids: np.ndarray, target_addrs: np.ndarray, target_ids: np.ndarray
    ) -> Dict[str, Set[str]]:
        """
        Match entities sharing both a numeric address component (e.g. house/street/pin code)
        and an address token.
        """
        # Build inverted index keyed by numeric tokens (length >= 2)
        number_index = defaultdict(lambda: defaultdict(list))
        num_pattern = re.compile(r"\b\d{2,}\b")

        for addr, tid in zip(target_addrs, target_ids):
            if not addr:
                continue
            nums = num_pattern.findall(addr)
            if not nums:
                continue
            tokens = [t for t in addr.split() if len(t) >= self.min_addr_token_len and not t.isdigit()]
            for n in set(nums):
                for tok in set(tokens):
                    number_index[n][tok].append(tid)

        candidates = defaultdict(set)
        for s1_id, addr in zip(s1_ids, s1_addrs):
            if not addr:
                continue
            nums = num_pattern.findall(addr)
            if not nums:
                continue
            tokens = [t for t in addr.split() if len(t) >= self.min_addr_token_len and not t.isdigit()]
            for n in set(nums):
                if n in number_index:
                    sub_dict = number_index[n]
                    for tok in set(tokens):
                        if tok in sub_dict:
                            candidates[s1_id].update(sub_dict[tok])

        return candidates

    # -------------------------------------------------------------
    # PASS 4: Character TF-IDF Cosine Retrieval for Names
    # -------------------------------------------------------------
    def pass4_tfidf_names(
        self, s1_names: np.ndarray, s1_ids: np.ndarray, target_names: np.ndarray, target_ids: np.ndarray
    ) -> Dict[str, Set[str]]:
        """Retrieve top-K candidates using character 3-4 gram TF-IDF cosine similarity."""
        if len(s1_names) == 0 or len(target_names) == 0:
            return defaultdict(set)

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.cfg.get("tfidf_ngram_range", (3, 4)),
            min_df=self.cfg.get("tfidf_min_df", 2),
            max_features=self.cfg.get("tfidf_max_features", 50_000),
            dtype=np.float32,
        )

        try:
            target_matrix = vectorizer.fit_transform(target_names)
            s1_matrix = vectorizer.transform(s1_names)
        except ValueError:
            # Handles empty vocabulary if strings are too short
            return defaultdict(set)

        candidates = defaultdict(set)
        batch_size = 2000
        n_queries = s1_matrix.shape[0]

        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            s1_batch = s1_matrix[start_idx:end_idx]
            sim_matrix = s1_batch.dot(target_matrix.T).toarray()  # (batch_size, n_targets)

            for i, row in enumerate(sim_matrix):
                s1_id = s1_ids[start_idx + i]
                # Filter indices above threshold
                above_thresh = np.where(row >= self.tfidf_name_thresh)[0]
                if len(above_thresh) > 0:
                    # Sort top candidates
                    if len(above_thresh) > self.tfidf_name_k:
                        top_indices = above_thresh[np.argsort(-row[above_thresh])[: self.tfidf_name_k]]
                    else:
                        top_indices = above_thresh
                    candidates[s1_id].update(target_ids[top_indices])

        del vectorizer, target_matrix, s1_matrix
        gc.collect()
        return candidates

    # -------------------------------------------------------------
    # PASS 5: Character TF-IDF Cosine Retrieval for Addresses
    # -------------------------------------------------------------
    def pass5_tfidf_addrs(
        self, s1_addrs: np.ndarray, s1_ids: np.ndarray, target_addrs: np.ndarray, target_ids: np.ndarray
    ) -> Dict[str, Set[str]]:
        """Retrieve top-K candidates using character 3-4 gram TF-IDF on addresses."""
        # Only index records with non-empty addresses
        valid_target_mask = np.array([bool(a.strip()) for a in target_addrs])
        valid_s1_mask = np.array([bool(a.strip()) for a in s1_addrs])

        if not np.any(valid_target_mask) or not np.any(valid_s1_mask):
            return defaultdict(set)

        filtered_target_addrs = target_addrs[valid_target_mask]
        filtered_target_ids = target_ids[valid_target_mask]

        filtered_s1_addrs = s1_addrs[valid_s1_mask]
        filtered_s1_ids = s1_ids[valid_s1_mask]

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.cfg.get("tfidf_ngram_range", (3, 4)),
            min_df=self.cfg.get("tfidf_min_df", 2),
            max_features=self.cfg.get("tfidf_max_features", 50_000),
            dtype=np.float32,
        )

        try:
            target_matrix = vectorizer.fit_transform(filtered_target_addrs)
            s1_matrix = vectorizer.transform(filtered_s1_addrs)
        except ValueError:
            return defaultdict(set)

        candidates = defaultdict(set)
        batch_size = 2000
        n_queries = s1_matrix.shape[0]

        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            s1_batch = s1_matrix[start_idx:end_idx]
            sim_matrix = s1_batch.dot(target_matrix.T).toarray()

            for i, row in enumerate(sim_matrix):
                s1_id = filtered_s1_ids[start_idx + i]
                above_thresh = np.where(row >= self.tfidf_addr_thresh)[0]
                if len(above_thresh) > 0:
                    if len(above_thresh) > self.tfidf_addr_k:
                        top_indices = above_thresh[np.argsort(-row[above_thresh])[: self.tfidf_addr_k]]
                    else:
                        top_indices = above_thresh
                    candidates[s1_id].update(filtered_target_ids[top_indices])

        del vectorizer, target_matrix, s1_matrix
        gc.collect()
        return candidates

    # -------------------------------------------------------------
    # MASTER MULTI-PASS CANDIDATE GENERATION
    # -------------------------------------------------------------
    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        """
        Execute all 5 blocking passes across country partitions (open-set).
        Merges candidates and deduplicates.

        Returns
        -------
        candidates : Dict[str, Set[str]]
            Mapping of source1_entity_id -> set of candidate entity IDs (from S2 and S3).
        stats : Dict[str, int]
            Candidate counts per pass and total deduplicated candidate count.
        """
        # Combine S2 and S3 as target pool
        target_df = pd.concat([s2_df, s3_df], ignore_index=True)

        final_candidates = {s1_id: set() for s1_id in s1_df["entity_id"].values}
        pass_counts = {
            "pass1_exact_name": 0,
            "pass2_name_tokens": 0,
            "pass3_address_tokens": 0,
            "pass4_tfidf_names": 0,
            "pass5_tfidf_addrs": 0,
            "total_before_dedup": 0,
            "total_after_dedup": 0,
        }

        # Dynamic country grouping (open-set, works for US, India, France, or any unseen country)
        all_countries = sorted(list(s1_df["norm_country"].unique()))
        if verbose:
            print(f"Executing MultiPassBlocker across {len(all_countries)} country partitions: {all_countries}")

        for country in all_countries:
            s1_sub = s1_df[s1_df["norm_country"] == country]
            target_sub = target_df[target_df["norm_country"] == country]

            if len(s1_sub) == 0 or len(target_sub) == 0:
                continue

            s1_ids = s1_sub["entity_id"].values
            s1_names = s1_sub["norm_name"].values
            s1_addrs = s1_sub["norm_address"].values

            target_ids = target_sub["entity_id"].values
            target_names = target_sub["norm_name"].values
            target_addrs = target_sub["norm_address"].values

            if verbose:
                print(f"\n--- Country [{country}]: S1 = {len(s1_sub):,}, Targets (S2+S3) = {len(target_sub):,} ---")

            # PASS 1: Exact Name
            p1 = self.pass1_exact_name(s1_names, s1_ids, target_names, target_ids)
            c1_cnt = sum(len(v) for v in p1.values())
            pass_counts["pass1_exact_name"] += c1_cnt
            if verbose:
                print(f"  Pass 1 (Exact Name): {c1_cnt:,} candidates")

            # PASS 2: Name Tokens / Prefixes
            p2 = self.pass2_name_tokens(s1_names, s1_ids, target_names, target_ids)
            c2_cnt = sum(len(v) for v in p2.values())
            pass_counts["pass2_name_tokens"] += c2_cnt
            if verbose:
                print(f"  Pass 2 (Name Tokens): {c2_cnt:,} candidates")

            # PASS 3: Address Tokens & Numbers
            p3 = self.pass3_address_tokens(s1_addrs, s1_ids, target_addrs, target_ids)
            c3_cnt = sum(len(v) for v in p3.values())
            pass_counts["pass3_address_tokens"] += c3_cnt
            if verbose:
                print(f"  Pass 3 (Address Tokens): {c3_cnt:,} candidates")

            # PASS 4: Character TF-IDF Names
            p4 = self.pass4_tfidf_names(s1_names, s1_ids, target_names, target_ids)
            c4_cnt = sum(len(v) for v in p4.values())
            pass_counts["pass4_tfidf_names"] += c4_cnt
            if verbose:
                print(f"  Pass 4 (TF-IDF Names): {c4_cnt:,} candidates")

            # PASS 5: Character TF-IDF Addresses
            p5 = self.pass5_tfidf_addrs(s1_addrs, s1_ids, target_addrs, target_ids)
            c5_cnt = sum(len(v) for v in p5.values())
            pass_counts["pass5_tfidf_addrs"] += c5_cnt
            if verbose:
                print(f"  Pass 5 (TF-IDF Addresses): {c5_cnt:,} candidates")

            # Union of all passes for this country
            for s1_id in s1_ids:
                merged = (
                    p1.get(s1_id, set())
                    | p2.get(s1_id, set())
                    | p3.get(s1_id, set())
                    | p4.get(s1_id, set())
                    | p5.get(s1_id, set())
                )
                # Keep only valid S2- and S3- prefixes
                valid_merged = {m for m in merged if m.startswith(("S2-", "S3-"))}

                # Cap max candidates per S1 entity if exceeded
                if len(valid_merged) > self.max_candidates:
                    valid_merged = set(list(valid_merged)[: self.max_candidates])

                final_candidates[s1_id].update(valid_merged)

        # Compute overall statistics
        pass_counts["total_before_dedup"] = (
            pass_counts["pass1_exact_name"]
            + pass_counts["pass2_name_tokens"]
            + pass_counts["pass3_address_tokens"]
            + pass_counts["pass4_tfidf_names"]
            + pass_counts["pass5_tfidf_addrs"]
        )
        pass_counts["total_after_dedup"] = sum(len(v) for v in final_candidates.values())

        if verbose:
            print("\n==========================================")
            print("Blocking Summary:")
            print(f"  - Total candidates before deduplication: {pass_counts['total_before_dedup']:,}")
            print(f"  - Total candidates after deduplication:  {pass_counts['total_after_dedup']:,}")
            reduction_from_raw = (
                1.0 - (pass_counts["total_after_dedup"] / (pass_counts["total_before_dedup"] + 1e-9))
            ) * 100
            print(f"  - Multi-pass overlap reduction:          {reduction_from_raw:.2f}%")
            print("==========================================")

        return final_candidates, pass_counts
