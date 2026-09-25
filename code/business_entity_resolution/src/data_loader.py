"""
Robust TSV data loader and validation module for Amazon ML Challenge 2026.
Ensures strict schema validation, duplicate ID checking, and tab-separated formats.
"""

import os
from pathlib import Path
from typing import Dict, Set, Optional, Tuple, Union
import pandas as pd

from .config import (
    SOURCE_COLUMNS,
    GT_COLUMNS,
    TRAIN_S1_PATH,
    TRAIN_S2_PATH,
    TRAIN_S3_PATH,
    TRAIN_GT_PATH,
    TEST_S1_PATH,
    TEST_S2_PATH,
    TEST_S3_PATH,
)


def validate_file_exists(filepath: Union[str, Path]) -> Path:
    """Ensure a required file exists on disk."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"Required dataset file not found: {p.resolve()}")
    return p


def load_source_file(
    filepath: Union[str, Path],
    expected_prefix: Optional[str] = None,
    validate_duplicates: bool = True,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a business record TSV file and enforce strict validation.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the TSV file.
    expected_prefix : str, optional
        Expected entity_id prefix ('S1-', 'S2-', 'S3-').
    validate_duplicates : bool
        Whether to verify all entity_id values are strictly unique.
    nrows : int, optional
        Optional row limit for quick testing.
        
    Returns
    -------
    pd.DataFrame with columns: entity_id, business_name, business_address, country
    """
    p = validate_file_exists(filepath)
    
    # Read strictly with sep="\t" and string dtypes
    df = pd.read_csv(
        p,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )
    
    # Validate columns
    actual_cols = list(df.columns)
    if actual_cols != SOURCE_COLUMNS:
        raise ValueError(
            f"Schema mismatch in {p.name}!\nExpected: {SOURCE_COLUMNS}\nFound: {actual_cols}"
        )
        
    # Validate entity_id prefix
    if expected_prefix:
        non_matching_mask = ~df["entity_id"].str.startswith(expected_prefix)
        if non_matching_mask.any():
            invalid_samples = df.loc[non_matching_mask, "entity_id"].head(5).tolist()
            raise ValueError(
                f"File {p.name} contains entity IDs not starting with expected prefix '{expected_prefix}': {invalid_samples}"
            )
            
    # Validate unique entity IDs
    if validate_duplicates:
        n_rows = len(df)
        n_unique = df["entity_id"].nunique()
        if n_unique != n_rows:
            dup_ids = df[df["entity_id"].duplicated(keep=False)]["entity_id"].head(5).tolist()
            raise ValueError(
                f"Duplicate entity IDs found in {p.name}! Total rows: {n_rows:,}, Unique IDs: {n_unique:,}. Examples: {dup_ids}"
            )
            
    return df


def load_ground_truth(
    filepath: Union[str, Path] = TRAIN_GT_PATH,
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """
    Load ground truth mapping from TSV file.
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['source1_entity_id', 'matched_entity_ids']
    gt_map : Dict[str, Set[str]]
        Dictionary mapping source1_entity_id -> set of matched entity IDs
    """
    p = validate_file_exists(filepath)
    
    df = pd.read_csv(
        p,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )
    
    actual_cols = list(df.columns)
    if actual_cols != GT_COLUMNS:
        raise ValueError(
            f"Schema mismatch in ground truth {p.name}!\nExpected: {GT_COLUMNS}\nFound: {actual_cols}"
        )
        
    # Validate uniqueness of source1_entity_id
    n_rows = len(df)
    n_unique = df["source1_entity_id"].nunique()
    if n_unique != n_rows:
        dup_ids = df[df["source1_entity_id"].duplicated(keep=False)]["source1_entity_id"].head(5).tolist()
        raise ValueError(
            f"Duplicate source1_entity_id in ground truth file! Examples: {dup_ids}"
        )
        
    # Build dictionary mapping
    gt_map = {}
    s1_ids = df["source1_entity_id"].values
    matched_strs = df["matched_entity_ids"].values
    
    for s1, m_str in zip(s1_ids, matched_strs):
        m_str = m_str.strip()
        if not m_str:
            gt_map[s1] = set()
        else:
            gt_map[s1] = {x.strip() for x in m_str.split(",") if x.strip()}
            
    return df, gt_map


def load_train_sources(nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """Load all training source files and ground truth."""
    s1 = load_source_file(TRAIN_S1_PATH, expected_prefix="S1-", nrows=nrows)
    s2 = load_source_file(TRAIN_S2_PATH, expected_prefix="S2-", nrows=nrows)
    s3 = load_source_file(TRAIN_S3_PATH, expected_prefix="S3-", nrows=nrows)
    _, gt_map = load_ground_truth(TRAIN_GT_PATH, nrows=nrows)
    return s1, s2, s3, gt_map


def load_test_sources(nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all test source files."""
    s1 = load_source_file(TEST_S1_PATH, expected_prefix="S1-", nrows=nrows)
    s2 = load_source_file(TEST_S2_PATH, expected_prefix="S2-", nrows=nrows)
    s3 = load_source_file(TEST_S3_PATH, expected_prefix="S3-", nrows=nrows)
    return s1, s2, s3


def export_id_list_tsv(
    mapping: Dict[str, Set[str]],
    all_s1_ids: Union[list, set, pd.Series],
    output_path: Union[str, Path],
    col_name: str = "matched_entity_ids",
) -> None:
    """
    Export a mapping of {s1_id: set_of_ids} to the challenge's strict TSV format.
    Guarantees:
    - Every S1 entity has exactly one row.
    - Tab-separated.
    - Matched/candidate IDs are comma-separated without spaces.
    - No self-matches (S1-) or duplicates.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    # Preserve order or sort S1 IDs
    for s1_id in all_s1_ids:
        matched_set = mapping.get(s1_id, set())
        # Filter out any accidental S1 self-matches
        valid_ids = sorted([mid for mid in matched_set if mid.startswith(("S2-", "S3-"))])
        rows.append({
            "source1_entity_id": s1_id,
            col_name: ",".join(valid_ids)
        })
        
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_p, sep="\t", index=False)
    print(f"Exported {len(df_out):,} rows to {out_p.resolve()}")
