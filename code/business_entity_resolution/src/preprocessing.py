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


# Fast regex patterns
PUNCT_SPLIT = re.compile(r"[^a-zA-Z0-9]+", re.UNICODE)
DOMAIN_END = re.compile(r"\.(?:com|org|net|in|fr|co\.in|co|io)$", re.I)

LEGAL_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "llp", "plc", "gmbh", "ag", "sarl", "sas", "sa", "sci", "eurl", "snc", "co",
    "societe", "services", "enterprises", "solutions", "holdings"
}

# Clean address token map (token -> expanded)
ADDR_TOKEN_MAP = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "pkwy": "parkway",
    "hwy": "highway", "apt": "apartment", "ste": "suite", "fl": "floor", "bldg": "building",
    "opp": "opposite", "nr": "near"
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters using NFKD decomposition."""
    if not text:
        return ""
    return unicodedata.normalize("NFKD", text)


def normalize_business_name(name: Optional[str]) -> str:
    """
    High-speed, high-recall business name normalization:
    - Unicode NFKD normalization
    - Lowercase
    - Strip URLs / domains (.com, .org, etc.)
    - Replace '&' with 'and', '@' with 'at'
    - Split on punctuation
    - Strip legal entity tokens from start AND end (LLC, Inc, Pvt Ltd, SARL, SAS, etc.)
    - Handle bracketed legal noise and spaced acronyms (l l p, l l c)
    """
    if not name or pd.isna(name):
        return ""
    
    text = str(name).strip().lower()
    if not text:
        return ""
        
    text = normalize_unicode(text)
    
    if text.startswith(("http://", "https://")):
        text = text.split("://", 1)[1]
    if text.startswith("www."):
        text = text[4:]
    text = DOMAIN_END.sub("", text)
    
    text = text.replace("&", " and ").replace("@", " at ")
    toks = [t for t in PUNCT_SPLIT.split(text) if t]
    
    # Strip leading legal forms (e.g., [LLP] Swastik -> Swastik)
    start = 0
    while start < len(toks) and toks[start] in LEGAL_TOKENS:
        start += 1
        
    # Strip trailing legal forms (e.g., Apple Inc -> Apple)
    end = len(toks)
    while end > start and toks[end - 1] in LEGAL_TOKENS:
        end -= 1
        
    toks = toks[start:end]
    
    # Strip trailing single-letter legal acronyms (e.g., l l p, l l c, p v t)
    if len(toks) >= 4 and toks[-3:] in (["l", "l", "p"], ["l", "l", "c"], ["p", "v", "t"]):
        toks = toks[:-3]
        
    return " ".join(toks)


def normalize_business_address(address: Optional[str]) -> str:
    """
    High-speed, high-recall business address normalization:
    - Unicode NFKD normalization
    - Lowercase
    - Filter placeholder tokens (null, none, nan)
    - Expand standard abbreviations via fast O(1) token mapping (11x faster than regex)
    - Preserve numbers and vital address tokens
    - Whitespace normalization
    """
    if not address or pd.isna(address):
        return ""
    
    text = str(address).strip().lower()
    if not text:
        return ""
        
    text = normalize_unicode(text)
    
    # Token-based fast expansion and normalization
    toks = [
        ADDR_TOKEN_MAP.get(t, t)
        for t in PUNCT_SPLIT.split(text)
        if t and t not in ("null", "none", "nan")
    ]
    return " ".join(toks)


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
