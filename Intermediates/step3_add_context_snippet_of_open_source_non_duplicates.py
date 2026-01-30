import pandas as pd
import requests
import re
import time
import os
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

INPUT_FILE = "Step2_Output.csv"
OUTPUT_FILE = "step3_Output.csv"
LENS_API_KEY = ""
API_URL = "https://api.lens.org/patent/search"
BATCH_SIZE = 50 
CONTEXT_WINDOW = 1000
def get_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST", "GET"])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def aggressive_context_extract(full_text, accession_id, window=1000):
    """
    Robust extraction that handles:
    1. Decimals (CBS 280.96)
    2. Missing spaces (ATCC12345)
    3. Newlines/Tabs
    """
    if not full_text or not isinstance(full_text, str): return ""
    
    clean_target_id = re.sub(r'[^A-Z0-9\.]', '', str(accession_id).upper())
    
    normalized_text_list = []
    normalized_map = []
    
    for i, char in enumerate(full_text):
        if char.isalnum() or char == '.':
            normalized_text_list.append(char.upper())
            normalized_map.append(i)
            
    normalized_text = "".join(normalized_text_list)
    
    match_index = normalized_text.find(clean_target_id)
    
    if match_index != -1:
        try:
            start_map_idx = match_index
            end_map_idx = match_index + len(clean_target_id) - 1
            
            orig_start = normalized_map[start_map_idx]
            orig_end = normalized_map[end_map_idx]
            
            final_start = max(0, orig_start - window)
            final_end = min(len(full_text), orig_end + window)
            
            snippet = full_text[final_start:final_end]
            return snippet.replace("\n", " ").replace("\r", " ").strip()
        except IndexError:
            return ""
            
    return ""
def fetch_snippets():
    print(f"[*] Loading input: {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"[!] Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    if 'LIBERATED_STATUS' in df.columns:
        df = df[df['LIBERATED_STATUS'] == 'OPEN SOURCE'].copy()
    elif 'Found_In_Claims' in df.columns:
        df = df[df['Found_In_Claims'] == True].copy()
    
    print(f"[*] Total Liberated Assets to process: {len(df)}")

    processed_keys = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_done = pd.read_csv(OUTPUT_FILE, on_bad_lines='skip')
            processed_keys = set(df_done['Lens_ID'].astype(str) + "_" + df_done['Accession_ID'].astype(str))
            print(f"[*] Resuming: {len(processed_keys)} assets already finished.")
        except:
            print("[!] Warning: Could not read existing output. Will append.")
    else:
        pd.DataFrame(columns=["Accession_ID", "Repository", "Lens_ID", "Title", "Context_Snippet"]).to_csv(OUTPUT_FILE, index=False)
        print(f"[*] Created output file: {OUTPUT_FILE}")

    df['unique_key'] = df['Lens_ID'].astype(str) + "_" + df['Accession_ID'].astype(str)
    df_todo = df[~df['unique_key'].isin(processed_keys)].copy()
    
    if df_todo.empty:
        print("[*] Job Complete! All snippets extracted.")
        return

    print(f"[*] Remaining assets to fetch: {len(df_todo)}")
    
    unique_patent_ids = df_todo['Lens_ID'].unique()
    patent_batches = [unique_patent_ids[i:i + BATCH_SIZE] for i in range(0, len(unique_patent_ids), BATCH_SIZE)]
    
    session = get_session()
    headers = {"Authorization": f"Bearer {LENS_API_KEY}", "Content-Type": "application/json"}
    
    assets_saved_session = 0
    
    for i, batch_ids in enumerate(patent_batches):
        batch_results = []
        try:
            payload = {
                "query": {"terms": {"lens_id": list(batch_ids)}},
                "size": BATCH_SIZE,
                "include": ["lens_id", "description", "claims"]
            }
            
            response = session.post(API_URL, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                print(f"\n[!] API Error Batch {i}: {response.status_code}")
                time.sleep(5)
                continue
                
            data = response.json().get("data", [])
            
            text_map = {}
            for pat in data:
                desc = pat.get("description", {}).get("text", "")
                c_data = pat.get("claims", [])
                claims_str = str(c_data) if isinstance(c_data, list) else ""
                text_map[pat["lens_id"]] = desc + " " + claims_str
            
            relevant_rows = df_todo[df_todo['Lens_ID'].isin(batch_ids)]
            for _, row in relevant_rows.iterrows():
                lid = row['Lens_ID']
                acc_id = row['Accession_ID']
                full_text = text_map.get(lid, "")
                
                snippet = aggressive_context_extract(full_text, acc_id, window=CONTEXT_WINDOW)
                
                batch_results.append({
                    "Accession_ID": acc_id,
                    "Repository": row['Repository'],
                    "Lens_ID": lid,
                    "Title": row['Title'],
                    "Context_Snippet": snippet
                })
            
            if batch_results:
                pd.DataFrame(batch_results).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                assets_saved_session += len(batch_results)
            
            print(f" -> Batch {i+1}/{len(patent_batches)} done. Saved {assets_saved_session} snippets.", end='\r')
            sys.stdout.flush()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n[!] Error in batch {i}: {e}")
            time.sleep(5)

    print(f"\n\n[SUCCESS] Run Complete. Snippets saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_snippets()
