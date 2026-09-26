"""
Multi-pass candidate generation (blocking) module for Amazon ML Challenge 2026.
Generates candidates from Source 1 to Source 2 and Source 3 using complementary passes:
1. Exact normalized name blocking
2. Normalized name token and prefix blocking
3. Character n-gram TF-IDF similarity retrieval for names
4. Normalized address token and number overlap blocking
5. Character n-gram TF-IDF similarity retrieval for addresses
6. Country-aware partitioning with seamless open-set fallback (US, India, France, and unseen countries).

Prioritizes recall while maintaining high computational efficiency through priority ranking
and multi-pass evidence fusion.
"""

from collections import defaultdict, Counter
import gc
import re
from typing import Dict, Set, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import BLOCKING_CONFIG

# Common domain suffixes for URL-derived business names
DOMAIN_PATTERN = re.compile(r"\.(?:com|org|net|in|fr|co\.in|co|io)\b", re.I)
NUM_PATTERN = re.compile(r"\b\d{2,}\b")

# High-frequency generic stopwords that should not trigger single-token matches
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


class MultiPassBlocker:
    """
    High-recall, multi-pass candidate generator operating across data sources.
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or BLOCKING_CONFIG
        self.min_name_token_len = self.cfg.get("min_name_token_len", 3)
        self.min_addr_token_len = self.cfg.get("min_addr_token_len", 4)
        self.max_token_freq = self.cfg.get("max_token_frequency", 5000)
        self.tfidf_name_k = self.cfg.get("tfidf_name_top_k", 8)
        self.tfidf_name_thresh = self.cfg.get("tfidf_name_threshold", 0.38)
        self.tfidf_addr_k = self.cfg.get("tfidf_addr_top_k", 8)
        self.tfidf_addr_thresh = self.cfg.get("tfidf_addr_threshold", 0.35)
        self.max_candidates = self.cfg.get("max_candidates_per_s1", 35)

    # -------------------------------------------------------------
    # PASS 1: Exact Normalized Business Name
    # -------------------------------------------------------------
    def pass1_exact_name(
        self, s1_names: np.ndarray, s1_ids: np.ndarray, target_names: np.ndarray, target_ids: np.ndarray
    ) -> Dict[str, Set[str]]:
        """Match exact normalized business names and strip domain variations."""
        name_index = defaultdict(list)
        for name, tid in zip(target_names, target_ids):
            if name:
                name_index[name].append(tid)
                # Strip potential web domains (e.g., jgsreal.com -> jgsreal)
                stripped = DOMAIN_PATTERN.sub("", name).strip()
                if stripped and stripped != name:
                    name_index[stripped].append(tid)

        candidates = defaultdict(set)
        for s1_id, name in zip(s1_ids, s1_names):
            if not name:
                continue
            if name in name_index:
                candidates[s1_id].update(name_index[name])
            stripped = DOMAIN_PATTERN.sub("", name).strip()
            if stripped and stripped in name_index:
                candidates[s1_id].update(name_index[stripped])

        return candidates

    # -------------------------------------------------------------
    # PASS 2: Normalized Name Token & Prefix Overlap Blocking
    # -------------------------------------------------------------
    def pass2_name_tokens(
        self, s1_names: np.ndarray, s1_ids: np.ndarray, target_names: np.ndarray, target_ids: np.ndarray
    ) -> Dict[str, Set[str]]:
        """Match significant name tokens and distinctive 2-word prefixes."""
        token_index = defaultdict(list)
        prefix_index = defaultdict(list)

        for name, tid in zip(target_names, target_ids):
            if not name:
                continue
            words = name.split()
            tokens = [t for t in words if len(t) >= self.min_name_token_len and t not in GENERIC_NAME_STOPWORDS]
            for token in set(tokens):
                token_index[token].append(tid)

            # Distinctive 2-word prefix
            if len(words) >= 2:
                pfx = words[0] + " " + words[1]
                prefix_index[pfx].append(tid)

        # Filter out hyper-frequent tokens to avoid candidate explosion
        token_index = {
            t: ids for t, ids in token_index.items() if len(ids) <= self.max_token_freq
        }
        prefix_index = {
            p: ids for p, ids in prefix_index.items() if len(ids) <= self.max_token_freq
        }

        candidates = defaultdict(set)
        for s1_id, name in zip(s1_ids, s1_names):
            if not name:
                continue
            words = name.split()
            tokens = [t for t in words if len(t) >= self.min_name_token_len and t not in GENERIC_NAME_STOPWORDS]
            for token in set(tokens):
                if token in token_index:
                    candidates[s1_id].update(token_index[token])

            if len(words) >= 2:
                pfx = words[0] + " " + words[1]
                if pfx in prefix_index:
                    candidates[s1_id].update(prefix_index[pfx])

        return candidates

    # -------------------------------------------------------------
    # PASS 3: Character TF-IDF Cosine Retrieval for Names
    # -------------------------------------------------------------
    def pass3_tfidf_names(
        self, s1_names: np.ndarray, s1_ids: np.ndarray, target_names: np.ndarray, target_ids: np.ndarray
    ) -> Dict[str, Set[str]]:
        """Retrieve top-K candidates using character 3-4 gram TF-IDF cosine similarity."""
        if len(s1_names) == 0 or len(target_names) == 0:
            return defaultdict(set)

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.cfg.get("tfidf_ngram_range", (3, 4)),
            min_df=self.cfg.get("tfidf_min_df", 2),
            max_features=self.cfg.get("tfidf_max_features", 80_000),
            dtype=np.float32,
        )

        try:
            target_matrix = vectorizer.fit_transform(target_names)
            s1_matrix = vectorizer.transform(s1_names)
        except ValueError:
            return defaultdict(set)

        candidates = defaultdict(set)
        batch_size = 2000
        n_queries = s1_matrix.shape[0]

        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            s1_batch = s1_matrix[start_idx:end_idx]
            sim_sparse = s1_batch.dot(target_matrix.T)
            sim_sparse.data[sim_sparse.data < self.tfidf_name_thresh] = 0
            sim_sparse.eliminate_zeros()

            for r in range(s1_batch.shape[0]):
                start_ptr = sim_sparse.indptr[r]
                end_ptr = sim_sparse.indptr[r + 1]
                if start_ptr < end_ptr:
                    row_data = sim_sparse.data[start_ptr:end_ptr]
                    row_indices = sim_sparse.indices[start_ptr:end_ptr]
                    s1_id = s1_ids[start_idx + r]
                    if len(row_indices) > self.tfidf_name_k:
                        top_local = np.argsort(-row_data)[: self.tfidf_name_k]
                        candidates[s1_id].update(target_ids[row_indices[top_local]])
                    else:
                        candidates[s1_id].update(target_ids[row_indices])

        del vectorizer, target_matrix, s1_matrix
        gc.collect()
        return candidates

    # -------------------------------------------------------------
    # PASS 4: Address Token & Number Overlap Blocking
    # -------------------------------------------------------------
    def pass4_address_tokens(
        self, s1_addrs: np.ndarray, s1_ids: np.ndarray, target_addrs: np.ndarray, target_ids: np.ndarray
    ) -> Dict[str, Set[str]]:
        """
        Match entities sharing distinctive address tokens (street/area/city/zip)
        and address numbers.
        """
        token_index = defaultdict(list)
        number_index = defaultdict(list)

        for addr, tid in zip(target_addrs, target_ids):
            if not addr:
                continue
            words = addr.split()
            tokens = [
                t for t in words
                if len(t) >= self.min_addr_token_len and t not in GENERIC_ADDR_STOPWORDS and not t.isdigit()
            ]
            for tok in set(tokens):
                token_index[tok].append(tid)

            nums = NUM_PATTERN.findall(addr)
            for n in set(nums):
                number_index[n].append(tid)

        # Filter out hyper-frequent address tokens (e.g. state names or huge cities)
        token_index = {
            t: ids for t, ids in token_index.items() if len(ids) <= self.max_token_freq
        }
        number_index = {
            n: ids for n, ids in number_index.items() if len(ids) <= self.max_token_freq
        }

        candidates = defaultdict(set)
        for s1_id, addr in zip(s1_ids, s1_addrs):
            if not addr:
                continue
            words = addr.split()
            tokens = [
                t for t in words
                if len(t) >= self.min_addr_token_len and t not in GENERIC_ADDR_STOPWORDS and not t.isdigit()
            ]
            nums = NUM_PATTERN.findall(addr)

            # Match on distinctive address tokens
            for tok in set(tokens):
                if tok in token_index:
                    candidates[s1_id].update(token_index[tok])

            # Match on address numbers if present
            for n in set(nums):
                if n in number_index:
                    candidates[s1_id].update(number_index[n])

        return candidates

    # -------------------------------------------------------------
    # PASS 5: Character TF-IDF Cosine Retrieval for Addresses
    # -------------------------------------------------------------
    def pass5_tfidf_addrs(
        self, s1_addrs: np.ndarray, s1_ids: np.ndarray, target_addrs: np.ndarray, target_ids: np.ndarray
    ) -> Dict[str, Set[str]]:
        """Retrieve top-K candidates using character 3-4 gram TF-IDF on addresses."""
        valid_target_mask = np.array([bool(a and a.strip()) for a in target_addrs])
        valid_s1_mask = np.array([bool(a and a.strip()) for a in s1_addrs])

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
            max_features=self.cfg.get("tfidf_max_features", 80_000),
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
            sim_sparse = s1_batch.dot(target_matrix.T)
            sim_sparse.data[sim_sparse.data < self.tfidf_addr_thresh] = 0
            sim_sparse.eliminate_zeros()

            for r in range(s1_batch.shape[0]):
                start_ptr = sim_sparse.indptr[r]
                end_ptr = sim_sparse.indptr[r + 1]
                if start_ptr < end_ptr:
                    row_data = sim_sparse.data[start_ptr:end_ptr]
                    row_indices = sim_sparse.indices[start_ptr:end_ptr]
                    s1_id = filtered_s1_ids[start_idx + r]
                    if len(row_indices) > self.tfidf_addr_k:
                        top_local = np.argsort(-row_data)[: self.tfidf_addr_k]
                        candidates[s1_id].update(filtered_target_ids[row_indices[top_local]])
                    else:
                        candidates[s1_id].update(filtered_target_ids[row_indices])

        del vectorizer, target_matrix, s1_matrix
        gc.collect()
        return candidates

    # -------------------------------------------------------------
    # MASTER MULTI-PASS CANDIDATE GENERATION WITH PRIORITY RANKING
    # -------------------------------------------------------------
    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        """
        Execute all 5 blocking passes across country partitions (Pass 6: country-aware).
        Takes union of candidates, applies priority ranking evidence fusion, and deduplicates.

        Returns
        -------
        candidates : Dict[str, Set[str]]
            Mapping of source1_entity_id -> set of candidate entity IDs (from S2 and S3).
        stats : Dict[str, int]
            Candidate counts per pass and total deduplicated candidate count.
        """
        target_df = pd.concat([s2_df, s3_df], ignore_index=True)

        final_candidates: Dict[str, Set[str]] = {s1_id: set() for s1_id in s1_df["entity_id"].values}
        pass_counts = {
            "pass1_exact_name": 0,
            "pass2_name_tokens": 0,
            "pass3_tfidf_names": 0,
            "pass4_address_tokens": 0,
            "pass5_tfidf_addrs": 0,
            "total_before_dedup": 0,
            "total_after_dedup": 0,
        }

        all_countries = sorted(list(s1_df["norm_country"].unique()))
        if verbose:
            print(f"Executing MultiPassBlocker across {len(all_countries)} country partitions: {all_countries}")

        for country in all_countries:
            s1_sub = s1_df[s1_df["norm_country"] == country]
            # Targets in the same country OR targets with unspecified/empty country
            target_sub = target_df[
                (target_df["norm_country"] == country) | (target_df["norm_country"] == "")
            ]

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
                print(f"  Pass 1 (Exact Name):            {c1_cnt:,} candidates")

            # PASS 2: Name Tokens / Prefixes
            p2 = self.pass2_name_tokens(s1_names, s1_ids, target_names, target_ids)
            c2_cnt = sum(len(v) for v in p2.values())
            pass_counts["pass2_name_tokens"] += c2_cnt
            if verbose:
                print(f"  Pass 2 (Name Tokens / Prefix):   {c2_cnt:,} candidates")

            # PASS 3: Character TF-IDF Names
            p3 = self.pass3_tfidf_names(s1_names, s1_ids, target_names, target_ids)
            c3_cnt = sum(len(v) for v in p3.values())
            pass_counts["pass3_tfidf_names"] += c3_cnt
            if verbose:
                print(f"  Pass 3 (TF-IDF Char Names):      {c3_cnt:,} candidates")

            # PASS 4: Address Tokens & Numbers
            p4 = self.pass4_address_tokens(s1_addrs, s1_ids, target_addrs, target_ids)
            c4_cnt = sum(len(v) for v in p4.values())
            pass_counts["pass4_address_tokens"] += c4_cnt
            if verbose:
                print(f"  Pass 4 (Address Tokens & Nums):  {c4_cnt:,} candidates")

            # PASS 5: Character TF-IDF Addresses
            p5 = self.pass5_tfidf_addrs(s1_addrs, s1_ids, target_addrs, target_ids)
            c5_cnt = sum(len(v) for v in p5.values())
            pass_counts["pass5_tfidf_addrs"] += c5_cnt
            if verbose:
                print(f"  Pass 5 (TF-IDF Char Addresses):  {c5_cnt:,} candidates")

            # Evidence fusion and priority ranking across passes
            for s1_id in s1_ids:
                cand_scores = Counter()
                s1_p1 = p1.get(s1_id, set())
                s1_p2 = p2.get(s1_id, set())
                s1_p3 = p3.get(s1_id, set())
                s1_p4 = p4.get(s1_id, set())
                s1_p5 = p5.get(s1_id, set())

                for cid in s1_p1:
                    cand_scores[cid] += 10.0
                for cid in s1_p3:
                    cand_scores[cid] += 6.0
                for cid in s1_p5:
                    cand_scores[cid] += 5.0
                for cid in s1_p4:
                    cand_scores[cid] += 4.0
                for cid in s1_p2:
                    cand_scores[cid] += 2.0

                # Multi-evidence bonus: candidate matches on both name and address
                name_cands = s1_p1 | s1_p2 | s1_p3
                addr_cands = s1_p4 | s1_p5
                for cid in (name_cands & addr_cands):
                    cand_scores[cid] += 5.0

                # Select top-ranked candidates
                ranked = [
                    cid for cid, score in cand_scores.most_common(self.max_candidates)
                    if cid.startswith(("S2-", "S3-"))
                ]
                final_candidates[s1_id] = set(ranked)

        # PASS 6 Fallback: for any S1 entity that still has 0 candidates, attempt cross-country exact name lookup
        zero_cand_s1 = [sid for sid, cands in final_candidates.items() if len(cands) == 0]
        if zero_cand_s1:
            if verbose:
                print(f"\nPass 6 Fallback: {len(zero_cand_s1):,} S1 entities have 0 candidates in partition. Checking global exact names...")
            s1_fallback = s1_df[s1_df["entity_id"].isin(set(zero_cand_s1))]
            p_fallback = self.pass1_exact_name(
                s1_fallback["norm_name"].values,
                s1_fallback["entity_id"].values,
                target_df["norm_name"].values,
                target_df["entity_id"].values,
            )
            for sid, cids in p_fallback.items():
                valid = {c for c in cids if c.startswith(("S2-", "S3-"))}
                if valid:
                    final_candidates[sid] = set(list(valid)[: self.max_candidates])

        pass_counts["total_before_dedup"] = (
            pass_counts["pass1_exact_name"]
            + pass_counts["pass2_name_tokens"]
            + pass_counts["pass3_tfidf_names"]
            + pass_counts["pass4_address_tokens"]
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
