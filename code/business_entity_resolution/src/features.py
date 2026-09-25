"""
Pairwise feature engineering module for Amazon ML Challenge 2026.
Extracts rich, high-performance numerical similarity features between Source 1 records
and candidate records (Source 2 / Source 3) for XGBoost binary classification.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

# Regex for extracting numeric components (e.g. house/street/pin codes)
DIGIT_PATTERN = re.compile(r"\b\d{2,}\b")

# Canonical feature columns in strict order
FEATURE_COLUMNS = [
    # Name features
    "name_exact_orig",
    "name_exact_norm",
    "name_token_jaccard",
    "name_token_overlap",
    "name_token_count_diff",
    "name_len_diff",
    "name_len_ratio",
    "name_levenshtein_sim",
    "name_jaro_winkler",
    "name_fuzz_ratio",
    "name_fuzz_token_sort_ratio",
    "name_fuzz_token_set_ratio",
    "name_char_3gram_jaccard",
    "name_char_cosine",
    # Address features
    "addr_is_missing",
    "addr_both_missing",
    "addr_exact_orig",
    "addr_exact_norm",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_token_count_diff",
    "addr_len_diff",
    "addr_len_ratio",
    "addr_levenshtein_sim",
    "addr_jaro_winkler",
    "addr_fuzz_ratio",
    "addr_fuzz_token_sort_ratio",
    "addr_fuzz_token_set_ratio",
    "addr_char_3gram_jaccard",
    "addr_char_cosine",
    "addr_num_exact_match",
    "addr_num_jaccard",
    # Other / Cross-field features
    "country_exact_match",
    "name_addr_sim_mean",
    "name_addr_sim_mult",
    "candidate_source",
]


def _char_ngram_cosine(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram cosine similarity (proxy for sublinear char TF-IDF)."""
    if not s1 or not s2 or len(s1) < n or len(s2) < n:
        return 0.0
    v1 = Counter([s1[i : i + n] for i in range(len(s1) - n + 1)])
    v2 = Counter([s2[i : i + n] for i in range(len(s2) - n + 1)])
    dot = sum(v1[k] * v2[k] for k in v1 if k in v2)
    norm1 = math.sqrt(sum(x * x for x in v1.values()))
    norm2 = math.sqrt(sum(x * x for x in v2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def _char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    if not s1 or not s2 or len(s1) < n or len(s2) < n:
        return 0.0
    set1 = set([s1[i : i + n] for i in range(len(s1) - n + 1)])
    set2 = set([s2[i : i + n] for i in range(len(s2) - n + 1)])
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return inter / union if union > 0 else 0.0


def compute_pair_features(
    s1_rec: Dict[str, Any],
    cand_rec: Dict[str, Any],
) -> List[float]:
    """
    Compute all 36 pairwise similarity features for a single candidate pair.
    Returns a list of floats matching FEATURE_COLUMNS.
    """
    # ------------------ NAME FEATURES ------------------
    name1_orig = s1_rec.get("business_name", "") or ""
    name2_orig = cand_rec.get("business_name", "") or ""
    name1_norm = s1_rec.get("norm_name", "") or ""
    name2_norm = cand_rec.get("norm_name", "") or ""

    name_exact_orig = 1.0 if name1_orig and name1_orig == name2_orig else 0.0
    name_exact_norm = 1.0 if name1_norm and name1_norm == name2_norm else 0.0

    toks1 = name1_norm.split()
    toks2 = name2_norm.split()
    set_t1 = set(toks1)
    set_t2 = set(toks2)

    inter_tok = len(set_t1.intersection(set_t2))
    union_tok = len(set_t1.union(set_t2))
    min_tok = min(len(set_t1), len(set_t2))

    name_token_jaccard = inter_tok / union_tok if union_tok > 0 else 0.0
    name_token_overlap = inter_tok / min_tok if min_tok > 0 else 0.0
    name_token_count_diff = float(abs(len(toks1) - len(toks2)))

    len1, len2 = len(name1_norm), len(name2_norm)
    name_len_diff = float(abs(len1 - len2))
    name_len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0

    # RapidFuzz string metrics
    if name1_norm and name2_norm:
        name_levenshtein_sim = distance.Levenshtein.normalized_similarity(name1_norm, name2_norm)
        name_jaro_winkler = distance.JaroWinkler.similarity(name1_norm, name2_norm)
        name_fuzz_ratio = fuzz.ratio(name1_norm, name2_norm) / 100.0
        name_fuzz_token_sort_ratio = fuzz.token_sort_ratio(name1_norm, name2_norm) / 100.0
        name_fuzz_token_set_ratio = fuzz.token_set_ratio(name1_norm, name2_norm) / 100.0
        name_char_3gram_jaccard = _char_ngram_jaccard(name1_norm, name2_norm, n=3)
        name_char_cosine = _char_ngram_cosine(name1_norm, name2_norm, n=3)
    else:
        name_levenshtein_sim = 0.0
        name_jaro_winkler = 0.0
        name_fuzz_ratio = 0.0
        name_fuzz_token_sort_ratio = 0.0
        name_fuzz_token_set_ratio = 0.0
        name_char_3gram_jaccard = 0.0
        name_char_cosine = 0.0

    # ------------------ ADDRESS FEATURES ------------------
    addr1_orig = s1_rec.get("business_address", "") or ""
    addr2_orig = cand_rec.get("business_address", "") or ""
    addr1_norm = s1_rec.get("norm_address", "") or ""
    addr2_norm = cand_rec.get("norm_address", "") or ""

    addr1_empty = (len(addr1_norm.strip()) == 0)
    addr2_empty = (len(addr2_norm.strip()) == 0)
    addr_is_missing = 1.0 if (addr1_empty or addr2_empty) else 0.0
    addr_both_missing = 1.0 if (addr1_empty and addr2_empty) else 0.0

    if not addr1_empty and not addr2_empty:
        addr_exact_orig = 1.0 if addr1_orig == addr2_orig else 0.0
        addr_exact_norm = 1.0 if addr1_norm == addr2_norm else 0.0

        a_toks1 = addr1_norm.split()
        a_toks2 = addr2_norm.split()
        a_set1 = set(a_toks1)
        a_set2 = set(a_toks2)

        a_inter = len(a_set1.intersection(a_set2))
        a_union = len(a_set1.union(a_set2))
        a_min = min(len(a_set1), len(a_set2))

        addr_token_jaccard = a_inter / a_union if a_union > 0 else 0.0
        addr_token_overlap = a_inter / a_min if a_min > 0 else 0.0
        addr_token_count_diff = float(abs(len(a_toks1) - len(a_toks2)))

        alen1, alen2 = len(addr1_norm), len(addr2_norm)
        addr_len_diff = float(abs(alen1 - alen2))
        addr_len_ratio = min(alen1, alen2) / max(alen1, alen2) if max(alen1, alen2) > 0 else 0.0

        addr_levenshtein_sim = distance.Levenshtein.normalized_similarity(addr1_norm, addr2_norm)
        addr_jaro_winkler = distance.JaroWinkler.similarity(addr1_norm, addr2_norm)
        addr_fuzz_ratio = fuzz.ratio(addr1_norm, addr2_norm) / 100.0
        addr_fuzz_token_sort_ratio = fuzz.token_sort_ratio(addr1_norm, addr2_norm) / 100.0
        addr_fuzz_token_set_ratio = fuzz.token_set_ratio(addr1_norm, addr2_norm) / 100.0
        addr_char_3gram_jaccard = _char_ngram_jaccard(addr1_norm, addr2_norm, n=3)
        addr_char_cosine = _char_ngram_cosine(addr1_norm, addr2_norm, n=3)

        # Numbers / PIN / house number matching
        nums1 = set(DIGIT_PATTERN.findall(addr1_norm))
        nums2 = set(DIGIT_PATTERN.findall(addr2_norm))
        if nums1 and nums2:
            num_inter = len(nums1.intersection(nums2))
            addr_num_exact_match = 1.0 if nums1 == nums2 else (0.5 if num_inter > 0 else 0.0)
            addr_num_jaccard = num_inter / len(nums1.union(nums2))
        elif not nums1 and not nums2:
            addr_num_exact_match = 0.5
            addr_num_jaccard = 0.5
        else:
            addr_num_exact_match = 0.0
            addr_num_jaccard = 0.0
    else:
        # Default safe values when address is missing
        addr_exact_orig = 0.0
        addr_exact_norm = 0.0
        addr_token_jaccard = 0.0
        addr_token_overlap = 0.0
        addr_token_count_diff = 0.0
        addr_len_diff = 0.0
        addr_len_ratio = 0.0
        addr_levenshtein_sim = 0.0
        addr_jaro_winkler = 0.0
        addr_fuzz_ratio = 0.0
        addr_fuzz_token_sort_ratio = 0.0
        addr_fuzz_token_set_ratio = 0.0
        addr_char_3gram_jaccard = 0.0
        addr_char_cosine = 0.0
        addr_num_exact_match = 0.0
        addr_num_jaccard = 0.0

    # ------------------ OTHER / CROSS-FIELD FEATURES ------------------
    c1 = s1_rec.get("norm_country", "") or ""
    c2 = cand_rec.get("norm_country", "") or ""
    country_exact_match = 1.0 if c1 and c1 == c2 else 0.0

    # Combined name & address signals
    if addr_is_missing:
        name_addr_sim_mean = name_fuzz_token_sort_ratio
        name_addr_sim_mult = name_fuzz_token_sort_ratio * 0.5
    else:
        name_addr_sim_mean = (name_fuzz_token_sort_ratio + addr_fuzz_token_sort_ratio) / 2.0
        name_addr_sim_mult = name_fuzz_token_sort_ratio * addr_fuzz_token_sort_ratio

    cand_id = cand_rec.get("entity_id", "") or ""
    candidate_source = 2.0 if cand_id.startswith("S2-") else (3.0 if cand_id.startswith("S3-") else 0.0)

    return [
        name_exact_orig,
        name_exact_norm,
        name_token_jaccard,
        name_token_overlap,
        name_token_count_diff,
        name_len_diff,
        name_len_ratio,
        name_levenshtein_sim,
        name_jaro_winkler,
        name_fuzz_ratio,
        name_fuzz_token_sort_ratio,
        name_fuzz_token_set_ratio,
        name_char_3gram_jaccard,
        name_char_cosine,
        addr_is_missing,
        addr_both_missing,
        addr_exact_orig,
        addr_exact_norm,
        addr_token_jaccard,
        addr_token_overlap,
        addr_token_count_diff,
        addr_len_diff,
        addr_len_ratio,
        addr_levenshtein_sim,
        addr_jaro_winkler,
        addr_fuzz_ratio,
        addr_fuzz_token_sort_ratio,
        addr_fuzz_token_set_ratio,
        addr_char_3gram_jaccard,
        addr_char_cosine,
        addr_num_exact_match,
        addr_num_jaccard,
        country_exact_match,
        name_addr_sim_mean,
        name_addr_sim_mult,
        candidate_source,
    ]


def extract_features_for_pairs(
    candidate_pairs: List[Tuple[str, str]],
    s1_dict: Dict[str, Dict[str, Any]],
    target_dict: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Extract pairwise features for a list of (s1_id, candidate_id) tuples.
    Returns a pandas DataFrame with columns matching FEATURE_COLUMNS.
    """
    data = []
    for s1_id, cand_id in candidate_pairs:
        s1_rec = s1_dict.get(s1_id, {})
        cand_rec = target_dict.get(cand_id, {})
        feats = compute_pair_features(s1_rec, cand_rec)
        data.append(feats)

    df_feats = pd.DataFrame(data, columns=FEATURE_COLUMNS, dtype=np.float32)
    return df_feats
