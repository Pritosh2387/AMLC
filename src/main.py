import pandas as pd
import numpy as np
import os
import joblib
from src.blocking import create_blocking_candidates
from src.features import compute_features

TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"

def clean_text_vectorized(series):
    # Vectorized string cleaning
    # handle NaN
    s = series.fillna("").astype(str)
    # lower
    s = s.str.lower()
    # remove non-alphanumeric
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    # remove extra spaces
    s = s.str.replace(r'\s+', ' ', regex=True)
    # strip
    s = s.str.strip()
    return s

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading test data...")
    s1 = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t")
    
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    
    print("Cleaning text...")
    s1['clean_name'] = clean_text_vectorized(s1['business_name'])
    s1['clean_addr'] = clean_text_vectorized(s1['business_address'])
    s1['combined'] = s1['clean_name'] + " " + s1['clean_addr']
    
    pool['clean_name'] = clean_text_vectorized(pool['business_name'])
    pool['clean_addr'] = clean_text_vectorized(pool['business_address'])
    pool['combined'] = pool['clean_name'] + " " + pool['clean_addr']
    
    print("Loading model...")
    model = joblib.load('model.joblib')
    
    all_candidates = []
    all_matches = []
    
    countries = s1['country'].dropna().unique()
    if s1['country'].isna().any():
        countries = list(countries) + [None]
        
    for country in countries:
        if country is None:
            s1_sub = s1[s1['country'].isna()].reset_index(drop=True)
            pool_sub = pool[pool['country'].isna()].reset_index(drop=True)
        else:
            s1_sub = s1[s1['country'] == country].reset_index(drop=True)
            pool_sub = pool[pool['country'] == country].reset_index(drop=True)
            
        if s1_sub.empty: continue
        
        # Lower threshold for more candidates
        candidates_dict = create_blocking_candidates(s1_sub, pool_sub, country, tfidf_threshold=0.3, chunk_size=5000)
        
        pool_dict = pool_sub.set_index('entity_id').to_dict('index')
        
        for _, row in s1_sub.iterrows():
            s1_id = row['entity_id']
            cands = candidates_dict.get(s1_id, set())
            
            all_candidates.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': ",".join(cands)
            })
            
            matched = []
            if cands:
                X_feats = []
                cand_list = list(cands)
                for c in cand_list:
                    if c in pool_dict:
                        feat = compute_features(row, pool_dict[c])
                        X_feats.append(feat)
                
                if X_feats:
                    df_X = pd.DataFrame(X_feats)
                    probs = model.predict_proba(df_X)[:, 1]
                    
                    threshold = 0.5 
                    
                    for i, p in enumerate(probs):
                        if p > threshold:
                            matched.append(cand_list[i])
                            
            all_matches.append({
                'source1_entity_id': s1_id,
                'matched_entity_ids': ",".join(matched)
            })
            
    print("Formatting outputs...")
    cand_df = pd.merge(s1[['entity_id']], pd.DataFrame(all_candidates), left_on='entity_id', right_on='source1_entity_id', how='left')
    match_df = pd.merge(s1[['entity_id']], pd.DataFrame(all_matches), left_on='entity_id', right_on='source1_entity_id', how='left')
    
    cand_df['candidate_entity_ids'] = cand_df['candidate_entity_ids'].fillna("")
    match_df['matched_entity_ids'] = match_df['matched_entity_ids'].fillna("")
    
    cand_df = cand_df.drop(columns=['source1_entity_id']).rename(columns={'entity_id': 'source1_entity_id'})
    match_df = match_df.drop(columns=['source1_entity_id']).rename(columns={'entity_id': 'source1_entity_id'})
    
    cand_df.to_csv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), sep="\t", index=False)
    match_df.to_csv(os.path.join(OUTPUT_DIR, "matching_results.tsv"), sep="\t", index=False)
    print("Done! Files saved to output/")

if __name__ == "__main__":
    main()