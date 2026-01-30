import requests
import json
import pandas as pd
import re
import datetime
import time
import os
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Tuple, Generator

LENS_API_KEY = ""
API_URL = "https://api.lens.org/patent/search"
OUTPUT_FILE = "Step2_Output.csv"
BATCH_SIZE = 100

CUSTOM_IDA_PATTERNS = {
    "ATCC": [
        r"ATCC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(PTA-[\d]+)",
        r"ATCC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(CRL-[\d]+)",
        r"ATCC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(HB-[\d]+)",
        r"ATCC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(CCL-[\d]+)",
        r"ATCC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(\d{4,})",
    ],
    "ECACC": [r"ECACC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(\d{8})", r"ECACC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(V\d+)"],
    "DSMZ": [r"DSM\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(DSM\s?\d+)", r"DSM\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(\d{4,})"],
    "NRRL": [r"NRRL\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*((?:B|Y|RL)[- ]?\d+)", r"NRRL\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(\d{4,})"],
    "IPOD": [r"(FERM\s+BP[- ]?\d+)", r"(FERM\s+P[- ]?\d+)"],
    "CBS": [r"CBS\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(\d+(?:\.\d+)?)"],
    "CCTCC": [r"CCTCC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*((?:M|V)\s?2\d{5})", r"CCTCC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(\d{4,})"],
    "KCTC": [r"KCTC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(\d{4,})"],
    "MTCC": [r"MTCC\s+(?:Accession\s+)?(?:No\.?|Number)?\s*[:]?\s*(\d{3,})"]
}

STANDARD_IDA_ACRONYMS = [
    "CGMCC", "GDMCC", "KCCM", "KCLRF", "KACC", "CNCM", "Westerdijk", "BCCM", 
    "LMG", "IHEM", "MUCL", "IDAC", "NML", "MCC", "NAIMCC", "VKM", "VKPM", "CECT", 
    "BEA", "DBVPG", "IZSLER", "CBA", "NMI", "PCM", "IAFB", "KPD", "NCIMB", "NCTC", 
    "NCYC", "CCAP", "IMI", "NIBSC", "NCMA", "IVS", "CChRGM", "CCM", "VTTCC", 
    "NCAIM", "MSCL", "CCMM", "CM-CNRG", "MUM", "UCCCB", "CCY", "CCOS", "NBIMCC"
]

def compile_regex_patterns() -> Dict[str, List[str]]:
    patterns = CUSTOM_IDA_PATTERNS.copy()
    generic_template = r"\b({})\b[\s:]+(?:Accession\s+)?(?:No\.?|Number)?\s*([0-9][0-9\.\-]*)"
    for acronym in STANDARD_IDA_ACRONYMS:
        regex = generic_template.format(acronym)
        patterns[acronym] = [regex]
    return patterns

ALL_PATTERNS = compile_regex_patterns()

def get_lens_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def fetch_gold_patents() -> Generator[List[dict], None, None]:
    BIO_IPC_PATTERNS = ["C12*", "C07K*", "A61K*", "A01H*", "C12Q*", "C07H*", "A23L*"]
    
    DEPOSIT_KEYWORDS = ["Budapest Treaty", "International Depository Authority", "biological deposit", "culture collection"] + \
                       list(CUSTOM_IDA_PATTERNS.keys()) + STANDARD_IDA_ACRONYMS

    query_payload = {
        "query": {
            "bool": {
                "must": [
                    {"terms": {"legal_status.patent_status": ["EXPIRED", "LAPSED", "REVOKED", "CEASED"]}},
                    {"bool": {
                        "should": [{"wildcard": {"class_ipc.symbol": code}} for code in BIO_IPC_PATTERNS],
                        "minimum_should_match": 1
                    }},
                    {"bool": {
                        "should": [{"match_phrase": {"full_text": kw}} for kw in DEPOSIT_KEYWORDS], 
                        "minimum_should_match": 1
                    }}
                ]
            }
        },
        "size": BATCH_SIZE,
        "scroll": "2m", 
        "include": ["lens_id", "biblio", "claims", "description", "legal_status"]
    }
    
    headers = {"Authorization": f"Bearer {LENS_API_KEY}", "Content-Type": "application/json"}
    session = get_lens_session()
    
    print(f"[*] Initializing Extended Gold Standard Mining...")
    
    try:
        response = session.post(API_URL, json=query_payload, headers=headers, timeout=60)
        if response.status_code != 200:
            print(f"[!] Init Failed: {response.text}")
            return
            
        data = response.json()
        scroll_id = data.get("scroll_id")
        patents = data.get("data", [])
        
        if not patents:
            print("[*] No patents found.")
            return
            
        yield patents
        
        total_fetched = len(patents)
        print(f" -> Batch 1 done. Total: {total_fetched}", end='\r')

        while True:
            if not scroll_id: break
            
            try:
                resp = session.post(API_URL, json={"scroll_id": scroll_id, "scroll": "2m"}, headers=headers, timeout=60)
                if resp.status_code != 200:
                    if resp.status_code >= 500: 
                        time.sleep(5)
                        continue
                    break
                
                data = resp.json()
                patents = data.get("data", [])
                scroll_id = data.get("scroll_id")
                
                if not patents: break
                
                yield patents
                
                total_fetched += len(patents)
                print(f" -> Processed: {total_fetched} patents", end='\r')
                
            except Exception as e:
                print(f"\n[!] Error: {e}")
                time.sleep(10)
                
    except Exception as e:
        print(f"\n[!] Critical Error: {e}")

def extract_accession_ids(text: str) -> List[Tuple[str, str]]:
    found_deposits = []
    if not text: return []
    
    for repo, patterns in ALL_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.groups():
                    raw_id = match.group(match.lastindex)
                else:
                    raw_id = match.group(0)
                
                clean_id = raw_id.rstrip(".").strip().upper()
                if len(clean_id) > 2 and any(char.isdigit() for char in clean_id):
                    found_deposits.append((repo, clean_id))
    return list(set(found_deposits))

def get_claims_text_robust(claims_data: list) -> str:
    extracted_parts = []
    def traverse(item):
        if isinstance(item, str): extracted_parts.append(item)
        elif isinstance(item, list):
            for sub in item: traverse(sub)
        elif isinstance(item, dict):
            for key in ['claim_text', 'text', 'claim']:
                if key in item and item[key]: traverse(item[key])
            if 'claims' in item: traverse(item['claims'])
    traverse(claims_data)
    return " ".join(extracted_parts)

def is_liberated(claims_text: str, acc_id: str) -> bool:
    if not claims_text: return False
    def normalize(s): return re.sub(r'[^A-Z0-9\.\-]', '', s.upper())
    return normalize(acc_id) in normalize(claims_text)

def process_batch(patents: List[dict]) -> List[dict]:
    results = []
    for patent in patents:
        lens_id = patent.get("lens_id")
        title = patent.get("biblio", {}).get("invention_title", [{}])[0].get("text", "No Title")
        claims_text = get_claims_text_robust(patent.get("claims", []))
        desc_text = patent.get("description", {}).get("text", "") 
        full_text = desc_text + " " + claims_text
        
        candidates = extract_accession_ids(full_text)
        if not candidates: continue

        for repo, acc_id in candidates:
            in_claims = is_liberated(claims_text, acc_id)
            results.append({
                "Lens_ID": lens_id,
                "Title": title,
                "Repository": repo,
                "Accession_ID": acc_id,
                "LIBERATED_STATUS": "OPEN SOURCE" if in_claims else "Mentioned",
                "Found_In_Claims": in_claims
            })
    return results

if __name__ == "__main__":
    start_time = time.time()
    
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=["Lens_ID", "Title", "Repository", "Accession_ID", "LIBERATED_STATUS", "Found_In_Claims"]).to_csv(OUTPUT_FILE, index=False)
        print(f"[*] Created output file: {OUTPUT_FILE}")
    else:
        print(f"[*] Appending to: {OUTPUT_FILE}")

    total_extracted = 0
    liberated_count = 0
    
    for patent_batch in fetch_gold_patents():
        batch_results = process_batch(patent_batch)
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            df_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            total_extracted += len(df_batch)
            liberated_count += df_batch['Found_In_Claims'].sum()
            
            print(f"   + Extracted {len(df_batch)} deposits ({liberated_count} Liberated).", end='\r')
            sys.stdout.flush()

    print(f"\n\n==============================================")
    print(f"[COMPLETE] Gold Mining Finished.")
    print(f"Total Deposit Events: {total_extracted}")
    print(f"Total LIBERATED Assets: {liberated_count}")
    print(f"Data saved to: {OUTPUT_FILE}")
    print(f"Time: {(time.time() - start_time)/60:.1f} mins")
    print(f"==============================================")
