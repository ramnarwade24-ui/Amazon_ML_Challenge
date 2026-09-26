import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import re
import unicodedata

cache_dir = Path("Amazon_ML_Challenge/models/benchmark_cache")
s1_df = pd.read_csv(cache_dir / "sample_s1.tsv", sep="\t", keep_default_na=False)
gt_df = pd.read_csv(cache_dir / "sample_gt.tsv", sep="\t", keep_default_na=False)
targets_df = pd.read_csv(cache_dir / "sample_targets.tsv", sep="\t", keep_default_na=False)
gt_map = {row["source1_entity_id"]: {x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()} for _, row in gt_df.iterrows()}

sys.path.insert(0, "Amazon_ML_Challenge/code/business_entity_resolution")
from src.preprocessing import normalize_business_name, normalize_business_address
from src.metrics import evaluate_blocking_recall

# Indexing targets
name_index = defaultdict(lambda: defaultdict(list))
pfx_index = defaultdict(lambda: defaultdict(list))
token_index = defaultdict(lambda: defaultdict(list))
addr_token_index = defaultdict(lambda: defaultdict(list))
num_index = defaultdict(lambda: defaultdict(list))

GENERIC_NAME_STOPWORDS = {"and", "the", "of", "in", "at", "for", "to", "a", "an", "services", "solutions", "enterprises", "center", "centre", "group", "associates", "consultants", "holdings"}
GENERIC_ADDR_STOPWORDS = {"near", "opp", "opposite", "road", "street", "st", "rd", "ave", "avenue", "lane", "dr", "drive", "blvd", "boulevard", "floor", "fl", "suite", "ste", "apt", "bldg", "building", "null", "none", "nan", "city", "town", "post", "box"}
NUM_PATTERN = re.compile(r"\b\d{2,}\b")
DOMAIN_PATTERN = re.compile(r"\.(?:com|org|net|in|fr|co\.in|co|io)\b", re.I)

for _, r in targets_df.iterrows():
    c = r["country"].strip().upper()
    eid = r["entity_id"]
    n = normalize_business_name(r["business_name"])
    a = normalize_business_address(r["business_address"])
    
    if n:
        name_index[c][n].append(eid)
        stripped = DOMAIN_PATTERN.sub("", n).strip()
        if stripped and stripped != n:
            name_index[c][stripped].append(eid)
        words = n.split()
        if len(words) >= 2:
            pfx = words[0] + " " + words[1]
            pfx_index[c][pfx].append(eid)
        for tok in set(words):
            if len(tok) >= 3 and tok not in GENERIC_NAME_STOPWORDS:
                token_index[c][tok].append(eid)
    if a:
        a_words = a.split()
        for tok in set(a_words):
            if len(tok) >= 4 and tok not in GENERIC_ADDR_STOPWORDS and not tok.isdigit():
                addr_token_index[c][tok].append(eid)
        for num in set(NUM_PATTERN.findall(a)):
            num_index[c][num].append(eid)

for c in token_index:
    token_index[c] = {k: v for k, v in token_index[c].items() if len(v) <= 3000}
    addr_token_index[c] = {k: v for k, v in addr_token_index[c].items() if len(v) <= 3000}
    num_index[c] = {k: v for k, v in num_index[c].items() if len(v) <= 3000}

cands = {}
for _, r in s1_df.iterrows():
    sid = r["entity_id"]
    c = r["country"].strip().upper()
    n = normalize_business_name(r["business_name"])
    a = normalize_business_address(r["business_address"])
    scores = Counter()

    if n in name_index[c]:
        for tid in name_index[c][n]: scores[tid] += 10.0
    stripped = DOMAIN_PATTERN.sub("", n).strip()
    if stripped and stripped in name_index[c]:
        for tid in name_index[c][stripped]: scores[tid] += 10.0

    words = n.split()
    if len(words) >= 2:
        pfx = words[0] + " " + words[1]
        if pfx in pfx_index[c]:
            for tid in pfx_index[c][pfx]: scores[tid] += 5.0

    for tok in set(words):
        if tok in token_index[c]:
            for tid in token_index[c][tok]: scores[tid] += 2.0

    if a:
        a_words = a.split()
        for tok in set(a_words):
            if tok in addr_token_index[c]:
                for tid in addr_token_index[c][tok]: scores[tid] += 3.0
        for num in set(NUM_PATTERN.findall(a)):
            if num in num_index[c]:
                for tid in num_index[c][num]: scores[tid] += 3.0

    ranked = [tid for tid, score in scores.most_common(35) if tid.startswith(("S2-", "S3-"))]
    cands[sid] = set(ranked)

m = evaluate_blocking_recall(gt_map, cands)
rec = m["candidate_recall"] * 100
print(f"Fast Streaming Blocking Recall (cap=35): {rec:.2f}% (captured: {m['captured_true_links']}/{m['total_true_links']})")
print(f"Entity Partial Recall: {m['entity_partial_recall']*100:.2f}%")
print(f"Entity Complete Recall: {m['entity_complete_recall']*100:.2f}%")
print(f"Average cands per S1: {m['avg_candidates_per_s1']:.2f}")
