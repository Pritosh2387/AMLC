import os
import re
import unicodedata
from collections import defaultdict, Counter
import time
import ctypes

import pandas as pd
import numpy as np
from rapidfuzz import fuzz

BENCHMARK_ROWS = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(ROOT, "output")

STRICT_MATCH_THRESHOLD = 0.88
MARGIN_THRESHOLD = 0.05
MAX_TOKEN_POSTINGS = 100

RARE_NAME_THRESHOLD = 3
RARE_ADDRESS_THRESHOLD = 3

STOPWORDS = {
    "inc", "llc", "ltd", "limited", "corp", "corporation", "company", "co",
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "boulevard",
    "blvd", "india", "usa", "france", "us", "uk", "private", "pvt"
}

def get_memory_usage_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        if os.name == 'nt':
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_uint32),
                    ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            memory_counters = PROCESS_MEMORY_COUNTERS()
            ctypes.windll.psapi.GetProcessMemoryInfo(process_handle, ctypes.byref(memory_counters), ctypes.sizeof(memory_counters))
            return memory_counters.WorkingSetSize / (1024 * 1024)
        return 0.0

def normalize_text(value):
    if pd.isna(value) or not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text):
    return set(text.split())

def jaccard_similarity(text_a, text_b):
    a = tokenize(text_a)
    b = tokenize(text_b)
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def calculate_score(name1, name2, address1, address2):
    name_ratio = fuzz.ratio(name1, name2) / 100.0
    name_token_ratio = fuzz.token_set_ratio(name1, name2) / 100.0
    name_partial_ratio = fuzz.partial_ratio(name1, name2) / 100.0
    
    address_ratio = fuzz.ratio(address1, address2) / 100.0
    address_token_ratio = fuzz.token_set_ratio(address1, address2) / 100.0
    
    name_jaccard = jaccard_similarity(name1, name2)
    address_jaccard = jaccard_similarity(address1, address2)
    
    score = (
        0.35 * name_token_ratio
        + 0.20 * name_ratio
        + 0.10 * name_partial_ratio
        + 0.15 * address_ratio
        + 0.10 * address_token_ratio
        + 0.05 * name_jaccard
        + 0.05 * address_jaccard
    )
    return score

def load_and_index_pool():
    print(f"[{get_memory_usage_mb():.1f} MB] Loading S2 and S3...")
    s2_path = os.path.join(TEST_DIR, "test_source2.tsv")
    s3_path = os.path.join(TEST_DIR, "test_source3.tsv")
    
    pool_data = defaultdict(lambda: {"entity_ids": [], "names": [], "addresses": []})
    
    for path in [s2_path, s3_path]:
        print(f"Reading {os.path.basename(path)}...")
        chunk_iter = pd.read_csv(
            path, 
            sep="\t", 
            chunksize=100000, 
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype=str
        )
        for chunk in chunk_iter:
            chunk["name_norm"] = chunk["business_name"].apply(normalize_text)
            chunk["address_norm"] = chunk["business_address"].apply(normalize_text)
            
            for row in chunk.itertuples(index=False):
                if pd.isna(row.country):
                    continue
                c = row.country
                pool_data[c]["entity_ids"].append(row.entity_id)
                pool_data[c]["names"].append(row.name_norm)
                pool_data[c]["addresses"].append(row.address_norm)

    print(f"[{get_memory_usage_mb():.1f} MB] Building indexes per country...")
    
    country_indexes = {}
    for c, data in pool_data.items():
        name_index = defaultdict(list)
        address_index = defaultdict(list)
        token_index = defaultdict(list)
        prefix_index = defaultdict(list)
        token_frequency = Counter()
        prefix_frequency = Counter()
        
        names = data["names"]
        addresses = data["addresses"]
        
        for name in names:
            tokens = set(name.split())
            for t in tokens:
                if len(t) >= 3 and t not in STOPWORDS:
                    token_frequency[t] += 1
            if len(name) >= 6:
                prefix_frequency[name[:6]] += 1
                
        for idx, name in enumerate(names):
            if len(name) >= 4:
                name_index[name].append(idx)
                
            tokens = set(name.split())
            for t in tokens:
                if len(t) >= 3 and t not in STOPWORDS:
                    if token_frequency[t] <= MAX_TOKEN_POSTINGS:
                        token_index[t].append(idx)
                        
            if len(name) >= 6:
                p = name[:6]
                if prefix_frequency[p] <= MAX_TOKEN_POSTINGS:
                    prefix_index[p].append(idx)
                    
        for idx, address in enumerate(addresses):
            if len(address) >= 8:
                address_index[address].append(idx)
                
        country_indexes[c] = {
            "name_index": dict(name_index),
            "address_index": dict(address_index),
            "token_index": dict(token_index),
            "prefix_index": dict(prefix_index),
            "token_frequency": dict(token_frequency)
        }
        
    print(f"[{get_memory_usage_mb():.1f} MB] Pool loading and indexing complete.")
    return pool_data, country_indexes

def generate_candidates(s1_name, s1_address, indexes):
    candidates = set()
    
    name_index = indexes["name_index"]
    address_index = indexes["address_index"]
    token_index = indexes["token_index"]
    prefix_index = indexes["prefix_index"]
    token_frequency = indexes["token_frequency"]
    
    if len(s1_name) >= 4:
        cands = name_index.get(s1_name, [])
        if len(cands) <= MAX_TOKEN_POSTINGS:
            candidates.update(cands)
            
    if len(s1_address) >= 8:
        cands = address_index.get(s1_address, [])
        if len(cands) <= MAX_TOKEN_POSTINGS:
            candidates.update(cands)
            
    if len(s1_name) >= 6:
        p = s1_name[:6]
        cands = prefix_index.get(p, [])
        if len(cands) <= min(MAX_TOKEN_POSTINGS, 100):
            candidates.update(cands)
            
    tokens = [t for t in set(s1_name.split()) if len(t) >= 3 and t not in STOPWORDS and t in token_index]
    tokens.sort(key=lambda t: token_frequency.get(t, 0))
    
    if tokens:
        t = tokens[0]
        cands = token_index.get(t, [])
        if len(cands) <= MAX_TOKEN_POSTINGS:
            candidates.update(cands)
            
    return candidates

def count_data_rows(path):
    """Count data rows in an existing TSV without loading it into memory."""
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        lines = sum(1 for _ in f)
    return max(0, lines - 1)


def process_s1():
    pool_data, country_indexes = load_and_index_pool()

    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resume automatically from the number of complete data rows already
    # present in BOTH output files. This preserves the first part of the
    # previous run instead of overwriting it.
    existing_candidate_rows = count_data_rows(candidate_path)
    existing_matching_rows = count_data_rows(matching_path)

    if existing_candidate_rows != existing_matching_rows:
        raise RuntimeError(
            "Output files are out of sync: "
            f"candidate_pairs.tsv has {existing_candidate_rows:,} data rows, "
            f"matching_results.tsv has {existing_matching_rows:,} data rows."
        )

    total_s1_rows = count_data_rows(s1_path)
    resume_rows = existing_candidate_rows

    if resume_rows > total_s1_rows:
        raise RuntimeError(
            f"Existing output has {resume_rows:,} rows, but S1 has only "
            f"{total_s1_rows:,} rows."
        )

    if resume_rows == total_s1_rows:
        print("\nAll S1 rows are already processed.")
        print(f"Rows processed: {total_s1_rows:,}/{total_s1_rows:,}")
        print(f"Candidate file:\n{candidate_path}")
        print(f"Matching file:\n{matching_path}")
        return

    print(
        f"\nResuming S1 processing from row {resume_rows + 1:,} "
        f"of {total_s1_rows:,}..."
    )
    print(f"Remaining rows: {total_s1_rows - resume_rows:,}")

    chunk_size = 25000
    rows_processed = resume_rows
    newly_processed = 0

    total_candidates_new = 0
    max_candidates_new = 0
    total_matches_new = 0
    max_matches_new = 0
    matched_s1_rows_new = 0

    dist = {"0": 0, "1-5": 0, "6-10": 0, "11-20": 0, "21-50": 0, "51-100": 0, ">100": 0}

    start_time = time.time()

    # The file has a header. Skip the header + already processed data rows,
    # then supply the known column names because the next line is data.
    s1_columns = ["entity_id", "business_name", "business_address", "country"]
    chunk_iter = pd.read_csv(
        s1_path,
        sep="\t",
        chunksize=chunk_size,
        skiprows=resume_rows + 1,
        header=None,
        names=s1_columns,
        dtype=str,
    )

    append_mode = "a" if resume_rows > 0 else "w"

    # Open once and append each completed chunk. This avoids repeatedly
    # opening the files for every chunk.
    with open(candidate_path, append_mode, encoding="utf-8") as fc, \
         open(matching_path, append_mode, encoding="utf-8") as fm:

        if resume_rows == 0:
            fc.write("source1_entity_id\tcandidate_entity_ids\n")
            fm.write("source1_entity_id\tmatched_entity_ids\n")

        for chunk in chunk_iter:
            chunk["name_norm"] = chunk["business_name"].apply(normalize_text)
            chunk["address_norm"] = chunk["business_address"].apply(normalize_text)

            candidates_out = []
            matches_out = []

            for row in chunk.itertuples(index=False):
                s1_id = row.entity_id
                s1_name = row.name_norm
                s1_address = row.address_norm
                c = row.country if not pd.isna(row.country) else None

                if c not in pool_data or c not in country_indexes:
                    candidates_out.append(f"{s1_id}\t\n")
                    matches_out.append(f"{s1_id}\t\n")
                    continue

                indexes = country_indexes[c]
                c_data = pool_data[c]

                name_index = indexes["name_index"]
                address_index = indexes["address_index"]

                candidate_indices = generate_candidates(
                    s1_name, s1_address, indexes
                )

                valid_candidates = []
                pool_ids = c_data["entity_ids"]
                pool_names = c_data["names"]
                pool_addresses = c_data["addresses"]

                for idx in candidate_indices:
                    c_id = pool_ids[idx]
                    if c_id.startswith("S2-") or c_id.startswith("S3-"):
                        valid_candidates.append(idx)

                candidate_ids = []
                scored_candidates = []

                for idx in valid_candidates:
                    c_id = pool_ids[idx]
                    candidate_ids.append(c_id)

                    c_name = pool_names[idx]
                    c_address = pool_addresses[idx]

                    name_freq = len(name_index.get(c_name, []))
                    addr_freq = len(address_index.get(c_address, []))

                    exact_name = bool(s1_name) and s1_name == c_name
                    exact_address = bool(s1_address) and s1_address == c_address

                    strong_name = exact_name and name_freq <= RARE_NAME_THRESHOLD
                    strong_addr = exact_address and addr_freq <= RARE_ADDRESS_THRESHOLD

                    # Keep the original scoring logic exactly the same.
                    score = calculate_score(
                        s1_name, c_name, s1_address, c_address
                    )

                    scored_candidates.append(
                        (c_id, score, strong_name, strong_addr)
                    )

                scored_candidates.sort(key=lambda x: x[1], reverse=True)

                accepted_matches = []

                if scored_candidates:
                    best_score = scored_candidates[0][1]
                    second_best_score = (
                        scored_candidates[1][1]
                        if len(scored_candidates) > 1
                        else 0.0
                    )
                    margin = best_score - second_best_score

                    for i, c_info in enumerate(scored_candidates):
                        c_id, score, strong_name, strong_addr = c_info
                        strong_evidence = strong_name or strong_addr

                        if strong_evidence:
                            accepted_matches.append(c_id)
                            continue

                        if score >= STRICT_MATCH_THRESHOLD:
                            if i == 0 and margin >= MARGIN_THRESHOLD:
                                accepted_matches.append(c_id)

                candidate_ids = list(dict.fromkeys(candidate_ids))
                matched_ids = list(dict.fromkeys(accepted_matches))

                candidate_count = len(candidate_ids)
                match_count = len(matched_ids)

                total_candidates_new += candidate_count
                max_candidates_new = max(max_candidates_new, candidate_count)
                total_matches_new += match_count
                max_matches_new = max(max_matches_new, match_count)

                if match_count:
                    matched_s1_rows_new += 1

                if match_count == 0:
                    dist["0"] += 1
                elif match_count <= 5:
                    dist["1-5"] += 1
                elif match_count <= 10:
                    dist["6-10"] += 1
                elif match_count <= 20:
                    dist["11-20"] += 1
                elif match_count <= 50:
                    dist["21-50"] += 1
                elif match_count <= 100:
                    dist["51-100"] += 1
                else:
                    dist[">100"] += 1

                candidates_out.append(
                    f"{s1_id}\t{','.join(candidate_ids)}\n"
                )
                matches_out.append(
                    f"{s1_id}\t{','.join(matched_ids)}\n"
                )

            fc.writelines(candidates_out)
            fm.writelines(matches_out)
            fc.flush()
            fm.flush()

            newly_processed += len(chunk)
            rows_processed += len(chunk)

            elapsed = time.time() - start_time
            rate = newly_processed / elapsed if elapsed > 0 else 0.0
            remaining = total_s1_rows - rows_processed
            eta_seconds = remaining / rate if rate > 0 else 0.0

            print(
                f"Progress: {rows_processed:,}/{total_s1_rows:,} "
                f"({rows_processed / total_s1_rows * 100:.1f}%) | "
                f"Rate: {rate:,.0f} rows/s | "
                f"ETA: {eta_seconds / 60:.1f} min",
                flush=True,
            )

            if (
                BENCHMARK_ROWS is not None
                and newly_processed >= BENCHMARK_ROWS
            ):
                break

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESUME RUN DONE")
    print("=" * 60)
    print(f"Previously processed: {resume_rows:,}")
    print(f"New rows processed: {newly_processed:,}")
    print(f"Total rows processed: {rows_processed:,}/{total_s1_rows:,}")
    print(f"Average candidates (new rows): {total_candidates_new / newly_processed:.2f}" if newly_processed else "Average candidates (new rows): 0.00")
    print(f"Max candidates (new rows): {max_candidates_new:,}")
    print(f"S1 rows with matches (new rows): {matched_s1_rows_new:,}")
    print(f"Total final matches (new rows): {total_matches_new:,}")
    print(f"Average final matches per S1 (new rows): {total_matches_new / newly_processed:.4f}" if newly_processed else "Average final matches per S1 (new rows): 0.0000")
    print(f"Max matches (new rows): {max_matches_new:,}")

    print("\nDistribution of final match counts (new rows):")
    for k, v in dist.items():
        print(f"  {k:8}: {v:,}")

    print(f"\nElapsed time: {elapsed:.2f} seconds")
    print(f"RAM usage: {get_memory_usage_mb():.1f} MB")
    print(f"\nCandidate file:\n{candidate_path}")
    print(f"Matching file:\n{matching_path}")

def main():
    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION")
    print("=" * 60)
    
    if BENCHMARK_ROWS is not None:
        print(f"*** BENCHMARK MODE: Processing first {BENCHMARK_ROWS} S1 rows ***")
        
    process_s1()

if __name__ == "__main__":
    main()
