"""
Text normalization module for Amazon ML Challenge 2026.
Normalizes business names and addresses without external lookups,
preserving open-set country flexibility (US, India, France, and any unseen countries).
"""

import re
import unicodedata
from typing import Optional
import pandas as pd


# Legal / Business entity suffixes for safe stripping / standardization at the end of names
# Supports English/US, Indian, and French legal forms
LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b("
    # English / International
    r"incorporated|inc\.?|corporation|corp\.?|limited\s+liability\s+company|llc\.?|l\.l\.c\.?|"
    r"private\s+limited|pvt\.?\s*ltd\.?|ltd\.?|limited|company|co\.?|co\s+ltd|"
    r"limited\s+liability\s+partnership|llp\.?|l\.l\.p\.?|plc\.?|p\.l\.c\.?|gmbh|ag|"
    # French legal entities
    r"societe\s+a\s+responsabilite\s+limitee|sarl\.?|s\.a\.r\.l\.?|"
    r"societe\s+par\s+actions\s+simplifiee|sas\.?|s\.a\.s\.?|"
    r"societe\s+anonyme|sa\.?|s\.a\.?|"
    r"societe\s+civile\s+immobiliere|sci\.?|s\.c\.i\.?|"
    r"entreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee|eurl\.?|e\.u\.r\.l\.?|"
    r"societe\s+en\s+nom\s+collectif|snc\.?|s\.n\.c\.?"
    r")\b\s*$",
    re.IGNORECASE,
)

# Common address token expansions for standard abbreviations
ADDRESS_ABBREVIATIONS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
}

ORDINAL_TYPO_REGEX = re.compile(r"(\d+)(nd|st|rd)\b")


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters using NFKD decomposition."""
    if not text:
        return ""
    return unicodedata.normalize("NFKD", text)


def normalize_business_name(name: Optional[str]) -> str:
    """
    Normalize business name:
    - Unicode normalization
    - Lowercase
    - Replace '&' with 'and', '@' with 'at'
    - Replace punctuation with spaces
    - Standardize and strip trailing legal suffixes
    - Collapse multiple spaces
    """
    if not name or pd.isna(name):
        return ""
    
    text = str(name).strip()
    if not text:
        return ""
        
    text = normalize_unicode(text).lower()
    
    # Standardize common connectors
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"@", " at ", text)
    
    # Strip URL prefixes/suffixes if present (e.g., maurewilliamscolombier.com -> maurewilliamscolombier)
    text = re.sub(r"^(?:https?:\/\/)?(?:www\.)?", "", text)
    text = re.sub(r"\.(?:com|org|net|in|fr|co\.in|co|io)\b", "", text)
    
    # Remove safe legal suffixes iteratively from end
    for _ in range(2):
        text = LEGAL_SUFFIXES_REGEX.sub("", text).strip()
        
    # Replace non-alphanumeric punctuation with spaces (preserves letters, numbers, and non-ASCII unicode)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_address(address: Optional[str]) -> str:
    """
    Normalize business address:
    - Unicode normalization
    - Lowercase
    - Filter out placeholder garbage tokens like 'null', 'none', 'nan'
    - Expand common road/street/building abbreviations
    - Preserve numbers and vital address tokens
    - Punctuation removal to space
    - Whitespace normalization
    """
    if not address or pd.isna(address):
        return ""
    
    text = str(address).strip()
    if not text:
        return ""
        
    text = normalize_unicode(text).lower()
    
    # Remove literal 'null' or 'nan' artifacts found in datasets
    text = re.sub(r"\b(null|none|nan)\b", " ", text)
    
    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    
    # Standardize street ordinals with typos (e.g., 45nd -> 45th)
    text = re.sub(r"(\d+)(?:nd|rd|st)\b", r"\1th", text)
    
    # Expand standard abbreviations
    for pattern, repl in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, repl, text)
        
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply name and address normalization to a DataFrame in-place or returning copy,
    preserving the original columns while adding 'norm_name' and 'norm_address'.
    """
    df = df.copy()
    
    # Add normalized name
    if "business_name" in df.columns:
        df["norm_name"] = df["business_name"].fillna("").astype(str).map(normalize_business_name)
    else:
        df["norm_name"] = ""
        
    # Add normalized address
    if "business_address" in df.columns:
        df["norm_address"] = df["business_address"].fillna("").astype(str).map(normalize_business_address)
    else:
        df["norm_address"] = ""
        
    # Clean country
    if "country" in df.columns:
        df["norm_country"] = df["country"].fillna("").astype(str).str.strip().str.upper()
    else:
        df["norm_country"] = ""
        
    return df
