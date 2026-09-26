import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import re, time

cache_dir = Path("Amazon_ML_Challenge/models/benchmark_cache")
s1_df = pd.read_csv(cache_dir / "sample_s1.tsv", sep="\t", keep_default_na=False)
gt_df = pd.read_csv(cache_dir / "sample_gt.tsv", sep="\t", keep_default_na=False)
targets_df = pd.read_csv(cache_dir / "sample_targets.tsv", sep="\t", keep_default_na=False)
gt_map = {row["source1_entity_id"]: {x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()} for _, row in gt_df.iterrows()}

sys.path.insert(0, "Amazon_ML_Challenge/code/business_entity_resolution")
from src.preprocessing import normalize_business_name, normalize_business_address
from src.metrics import evaluate_blocking_recall

t0 = time.time()
s1_exact = defaultdict(lambda: defaultdict(list))
s1_pfx = defaultdict(lambda: defaultdict(list))
s1_name_tok = defaultdict(lambda: defaultdict(list))
s1_addr_tok = defaultdict(lambda: defaultdict(list))
s1_num = defaultdict(lambda: defaultdict(list))

GENERIC_NAME_STOPWORDS = {"and", "the", "of", "in", "at", "for", "to", "a", "an", "services", "solutions", "enterprises", "center", "centre", "group", "associates", "consultants", "holdings"}
GENERIC_ADDR_STOPWORDS = {"near", "opp", "opposite", "road", "street", "st", "rd", "ave", "avenue", "lane", "dr", "drive", "blvd", "boulevard", "floor", "fl", "suite", "ste", "apt", "bldg", "building", "null", "none", "nan", "city", "town", "post", "box"}
NUM_PATTERN = re.compile(r"\b\d{2,}\b")
DOMAIN_PATTERN = re.compile(r"\.(?:com|org|net|in|fr|co\.in|co|io)\b", re.I)

s1_candidates = {sid: Counter() for sid in s1_df["entity_id"]}

for _, r in s1_df.iterrows():
    sid = r["entity_id"]
    c = r["country"].strip().upper()
    n = normalize_business_name(r["business_name"])
    a = normalize_business_address(r["business_address"])

    if n:
        s1_exact[c][n].append(sid)
        stripped = DOMAIN_PATTERN.sub("", n).strip()
        if stripped and stripped != n:
            s1_exact[c][stripped].append(sid)
        words = n.split()
        if len(words) >= 2:
            pfx = words[0] + " " + words[1]
            s1_pfx[c][pfx].append(sid)
        for tok in set(words):
            if len(tok) >= 3 and tok not in GENERIC_NAME_STOPWORDS:
                s1_name_tok[c][tok].append(sid)
    if a:
        a_words = a.split()
        for tok in set(a_words):
            if len(tok) >= 4 and tok not in GENERIC_ADDR_STOPWORDS and not tok.isdigit():
                s1_addr_tok[c][tok].append(sid)
        for num in set(NUM_PATTERN.findall(a)):
            s1_num[c][num].append(sid)

# Stream targets
for _, r in targets_df.iterrows():
    tid = r["entity_id"]
    c = r["country"].strip().upper()
    n = normalize_business_name(r["business_name"])
    a = normalize_business_address(r["business_address"])

    if n:
        if n in s1_exact[c]:
            for sid in s1_exact[c][n]: s1_candidates[sid][tid] += 10.0
        stripped = DOMAIN_PATTERN.sub("", n).strip()
        if stripped and stripped in s1_exact[c]:
            for sid in s1_exact[c][stripped]: s1_candidates[sid][tid] += 10.0

        words = n.split()
        if len(words) >= 2:
            pfx = words[0] + " " + words[1]
            if pfx in s1_pfx[c]:
                for sid in s1_pfx[c][pfx]: s1_candidates[sid][tid] += 5.0
        for tok in set(words):
            if tok in s1_name_tok[c]:
                for sid in s1_name_tok[c][tok]: s1_candidates[sid][tid] += 2.0

    if a:
        a_words = a.split()
        for tok in set(a_words):
            if tok in s1_addr_tok[c]:
                for sid in s1_addr_tok[c][tok]: s1_candidates[sid][tid] += 3.0
        for num in set(NUM_PATTERN.findall(a)):
            if num in s1_num[c]:
                for sid in s1_num[c][num]: s1_candidates[sid][tid] += 3.0

final_cands = {}
for sid, scores in s1_candidates.items():
    ranked = [tid for tid, score in scores.most_common(35) if tid.startswith(("S2-", "S3-"))]
    final_cands[sid] = set(ranked)

m = evaluate_blocking_recall(gt_map, final_cands)
print(f"Inverted Streaming Recall: {m['candidate_recall']*100:.2f}% (captured: {m['captured_true_links']}/{m['total_true_links']})")
print(f"Entity Partial Recall: {m['entity_partial_recall']*100:.2f}%")
print(f"Elapsed: {time.time()-t0:.2f}s")
