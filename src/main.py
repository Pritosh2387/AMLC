"""
FAST BLOCKING V2 - RECALL EVALUATOR (Memory & CPU Optimized)
Unstop 2026 Business Entity Resolution challenge.
"""

import argparse
import gc
import itertools
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "co", "company", "services", "solutions",
    "enterprises", "group", "holdings", "gmbh", "sarl", "sa", "pty", "plc",
    "llp", "lp"
}
STOPWORDS = {
    "the", "of", "and", "in", "at", "on", "for", "by", "with", "a", "an",
    "to", "from", "as", "is", "center", "centre", "plaza", "mall", "shop",
    "store", "express", "global", "national", "international"
}
ADDR_ABBR = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "ln": "lane", "dr": "drive", "ste": "suite",
    "apt": "apartment", "fl": "floor", "flr": "floor", "pkwy": "parkway",
    "hwy": "highway", "ct": "court", "pl": "place", "sq": "square"
}

# --- Optimized Normalization ---

PUNCT_RE = re.compile(r"[^a-z0-9\s]")
SPACE_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"\b\d+\b")

def clean_text(x):
    if not x or pd.isna(x): return ""
    s = unicodedata.normalize("NFKD", str(x)).encode('ascii', 'ignore').decode('utf-8')
    s = s.lower().replace("&", " and ")
    s = PUNCT_RE.sub(" ", s)
    return SPACE_RE.sub(" ", s).strip()

def norm_name(x):
    s = clean_text(x)
    if not s: return ""
    toks = [t for t in s.split() if t not in LEGAL_SUFFIXES]
    return " ".join(toks) if toks else s

def norm_addr(x):
    s = clean_text(x)
    if not s: return ""
    return " ".join(ADDR_ABBR.get(t, t) for t in s.split())

def norm_country(x):
    s = str(x).strip().upper() if x is not None else ""
    return {
        "USA": "US", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US",
        "IND": "IN", "INDIA": "IN", "FRA": "FR", "FRANCE": "FR"
    }.get(s, s or "UN")


# --- Fast Inverted Index V2 (Early Pruning) ---

B_NAME = 1
B_ADDR = 2
B_RARE = 3
B_PAIR = 4
B_NUM  = 5
B_SIG  = 6

BUCKET_NAMES = {
    B_NAME: "Exact-name",
    B_ADDR: "Exact-address",
    B_RARE: "Rare-token",
    B_PAIR: "Token-pair/trigram",
    B_NUM:  "Number+name",
    B_SIG:  "Character-signature"
}

# Caps array allows ultra-fast O(1) lookup during inner loop based on Bucket ID
# Format: (Pad, Name, Addr, Rare, Pair, Num, Sig)
CAPS = (0, 2000, 2000, 500, 800, 800, 200)

class FastIndexV2:
    def __init__(self, pool):
        self.ids = pool["entity_id"].to_numpy()
        self.names = pool["clean_name"].to_numpy()
        self.addrs = pool["clean_addr"].to_numpy()
        self.countries = pool["clean_country"].to_numpy()
        self.n = len(pool)
        self.id_to_idx = {x: i for i, x in enumerate(self.ids)}
        
        self.rare_vocab = set()
        self.index = {}
        self._build()

    def _build(self):
        print(f"[Index] Scanning {self.n:,} records for rare vocab...")
        t0 = time.time()
        
        df = Counter()
        for name in self.names:
            if not name: continue
            for t in set(name.split()):
                if len(t) >= 3 and t not in STOPWORDS and t not in LEGAL_SUFFIXES:
                    df[t] += 1
        
        self.rare_vocab = {t for t, c in df.items() if 2 <= c <= 150}
        del df
        print(f"[Index] Found {len(self.rare_vocab):,} rare tokens.")

        print("[Index] Extracting keys with Early-Pruning (Memory Safe)...")
        raw_idx = {}
        
        # Local refs for extreme speed inside the 10M loop
        names = self.names
        addrs = self.addrs
        countries = self.countries
        rare = self.rare_vocab
        
        for i in range(self.n):
            c = countries[i]
            n = names[i]
            a = addrs[i]
            
            keys = set()
            
            nt = []
            if n:
                keys.add((B_NAME, c, n))
                nt = [t for t in n.split() if t not in STOPWORDS and t not in LEGAL_SUFFIXES]
                
            at = []
            if a:
                keys.add((B_ADDR, c, a))
                at = [t for t in a.split() if t not in STOPWORDS and t not in LEGAL_SUFFIXES]
                
            for t in nt:
                if t in rare:
                    keys.add((B_RARE, c, t))
            
            if len(nt) >= 2:
                for p in itertools.combinations(nt[:4], 2):
                    keys.add((B_PAIR, c, p[0], p[1]))
            if len(nt) >= 3:
                for p in itertools.combinations(nt[:4], 3):
                    keys.add((B_PAIR, c, p[0], p[1], p[2]))
                    
            if a:
                nu = set(NUM_RE.findall(a))
                for num in nu:
                    for t in nt[:3]:
                        keys.add((B_NUM, c, num, t))
                    for t in at[:2]:
                        keys.add((B_NUM, c, num, t))
                        
            if n:
                n_ns = n.replace(" ", "")
                l = len(n_ns)
                if l >= 5:
                    keys.add((B_SIG, c, "pre5", n_ns[:5]))
                if l >= 7:
                    keys.add((B_SIG, c, "pre7", n_ns[:7]))
                    keys.add((B_SIG, c, "suf5", n_ns[-5:]))
                
                if nt:
                    first = nt[0]
                    keys.add((B_SIG, c, "ftok", first))
                    sig = "".join(sorted(set(first)))
                    if len(sig) >= 4:
                        keys.add((B_SIG, c, "fsig", sig))

            # EARLY PRUNING: Insert safely, blacklist immediately if cap exceeded
            for k in keys:
                val = raw_idx.get(k)
                if val is None:
                    raw_idx[k] = [i]
                elif val is not True:  # True = Blacklisted/Cap exceeded
                    val.append(i)
                    if len(val) > CAPS[k[0]]:
                        raw_idx[k] = True  # Instantly free memory!
                        
            if (i + 1) % 1_000_000 == 0:
                print(f"    Indexed {i + 1:,} / {self.n:,} rows...")
                
        print("[Index] Finalizing index and purging blacklisted keys...")
        # Strip all the 'True' values out, keeping only valid lists
        self.index = {k: v for k, v in raw_idx.items() if v is not True}
            
        del raw_idx
        gc.collect()
        print(f"[Index] Built in {time.time()-t0:.1f}s | Valid keys: {len(self.index):,}")

    def get_candidates(self, row):
        c = row["clean_country"]
        n = row["clean_name"]
        a = row["clean_addr"]
        
        keys = set()
        nt = []
        if n:
            keys.add((B_NAME, c, n))
            nt = [t for t in n.split() if t not in STOPWORDS and t not in LEGAL_SUFFIXES]
            
        at = []
        if a:
            keys.add((B_ADDR, c, a))
            at = [t for t in a.split() if t not in STOPWORDS and t not in LEGAL_SUFFIXES]
            
        for t in nt:
            if t in self.rare_vocab:
                keys.add((B_RARE, c, t))
        
        if len(nt) >= 2:
            for p in itertools.combinations(nt[:4], 2): keys.add((B_PAIR, c, p[0], p[1]))
        if len(nt) >= 3:
            for p in itertools.combinations(nt[:4], 3): keys.add((B_PAIR, c, p[0], p[1], p[2]))
                
        if a:
            nu = set(NUM_RE.findall(a))
            for num in nu:
                for t in nt[:3]: keys.add((B_NUM, c, num, t))
                for t in at[:2]: keys.add((B_NUM, c, num, t))
                    
        if n:
            n_ns = n.replace(" ", "")
            l = len(n_ns)
            if l >= 5: keys.add((B_SIG, c, "pre5", n_ns[:5]))
            if l >= 7:
                keys.add((B_SIG, c, "pre7", n_ns[:7]))
                keys.add((B_SIG, c, "suf5", n_ns[-5:]))
            if nt:
                first = nt[0]
                keys.add((B_SIG, c, "ftok", first))
                sig = "".join(sorted(set(first)))
                if len(sig) >= 4: keys.add((B_SIG, c, "fsig", sig))

        found = defaultdict(set)
        for k in keys:
            bucket = k[0]
            for pool_idx in self.index.get(k, []):
                found[pool_idx].add(bucket)
                
        return found


# --- Evaluation Runner ---

def load_and_clean_chunked(file_path, keep_raw=False):
    cols = ["entity_id", "business_name", "business_address", "country"]
    processed = []
    chunk_idx = 1
    
    for chunk in pd.read_csv(file_path, sep="\t", dtype=str, usecols=cols, chunksize=500_000):
        chunk = chunk.fillna("")
        chunk["clean_name"] = chunk["business_name"].map(norm_name)
        chunk["clean_addr"] = chunk["business_address"].map(norm_addr)
        chunk["clean_country"] = chunk["country"].map(norm_country)
        
        if not keep_raw:
            chunk.drop(columns=["business_name", "business_address", "country"], inplace=True)
            
        processed.append(chunk)
        print(f"    Loaded & cleaned chunk {chunk_idx} (500k rows)...")
        chunk_idx += 1
        gc.collect()
        
    return pd.concat(processed, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--val-size", type=int, default=25000, help="S1 records to evaluate")
    args = ap.parse_args()

    t0 = time.time()
    train_dir = Path(args.dataset_dir) / "train"

    print("="*70)
    print("FAST BLOCKING V2 - RECALL EVALUATOR (Memory Optimized)")
    print("="*70)

    print("\n[1/3] Loading and cleaning data in chunks...")
    
    print("  -> Processing train_source1.tsv...")
    s1 = load_and_clean_chunked(train_dir / "train_source1.tsv", keep_raw=True)
    
    print("  -> Processing train_source2.tsv (Pool)...")
    s2 = load_and_clean_chunked(train_dir / "train_source2.tsv", keep_raw=False)
    
    s3_path = train_dir / "train_source3.tsv"
    if s3_path.exists():
        print("  -> Processing train_source3.tsv (Pool)...")
        s3 = load_and_clean_chunked(s3_path, keep_raw=False)
        pool = pd.concat([s2, s3], ignore_index=True)
        del s3
    else:
        pool = s2
        
    del s2
    gc.collect()

    print("  -> Loading ground truth...")
    gt_df = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    gt = {}
    for sid, matches in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        gt[sid] = {x for x in matches.split(",") if x}
    del gt_df

    print(f"\n[2/3] Initializing FastIndexV2 over {len(pool):,} records...")
    index = FastIndexV2(pool)

    # Sample a validation set containing actual matches
    s1_with_matches = s1[s1["entity_id"].isin([k for k, v in gt.items() if v])].copy()
    val_s1 = s1_with_matches.sample(n=min(args.val_size, len(s1_with_matches)), random_state=SEED)
    
    val_records = val_s1.to_dict('records')
    print(f"\n[3/3] Evaluating blocking on {len(val_records):,} S1 validation records...")

    total_true_links = 0
    bucket_hits = {b: 0 for b in BUCKET_NAMES.keys()}
    cumulative_hits = set()
    cand_counts = []
    
    t_eval = time.time()
    for i, row in enumerate(val_records):
        sid = row["entity_id"]
        t_ids = gt.get(sid, set())
        t_indices = {index.id_to_idx[tid] for tid in t_ids if tid in index.id_to_idx}
        
        if not t_indices:
            continue
            
        total_true_links += len(t_indices)
        
        cand_map = index.get_candidates(row)
        cand_counts.append(len(cand_map))
        
        for t_idx in t_indices:
            found_buckets = cand_map.get(t_idx, set())
            for b in found_buckets:
                bucket_hits[b] += 1
            if found_buckets:
                cumulative_hits.add((sid, t_idx))
                
        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1:,} queries...")

    print(f"Evaluation complete in {time.time()-t_eval:.1f}s.\n")

    print("="*60)
    print("BLOCKING V2 EVALUATION REPORT")
    print("="*60)
    
    print(f"Total true links evaluated : {total_true_links:,}")
    print("\n--- INDEPENDENT RECALL BY STRATEGY ---")
    for b_id, b_name in sorted(BUCKET_NAMES.items()):
        hits = bucket_hits[b_id]
        rec = (hits / total_true_links) * 100 if total_true_links else 0
        print(f"{b_id}. {b_name:<25} : {rec:>6.2f}% ({hits:,} hits)")

    cum_recall = (len(cumulative_hits) / total_true_links) * 100 if total_true_links else 0
    print("\n--- CUMULATIVE ---")
    print(f"7. Cumulative union recall   : {cum_recall:>6.2f}% ({len(cumulative_hits):,}/{total_true_links:,})")

    avg_cands = np.mean(cand_counts) if cand_counts else 0
    p95 = np.percentile(cand_counts, 95) if cand_counts else 0
    p99 = np.percentile(cand_counts, 99) if cand_counts else 0
    max_cands = np.max(cand_counts) if cand_counts else 0

    print("\n--- CANDIDATE SET SIZES (Before any ML Truncation) ---")
    print(f"8. Average candidate count   : {avg_cands:,.2f}")
    print(f"9. p95 candidate count       : {p95:,.0f}")
    print(f"10. p99 candidate count      : {p99:,.0f}")
    print(f"11. Maximum candidates       : {max_cands:,.0f}")
    print("="*60)


if __name__ == "__main__":
    main()