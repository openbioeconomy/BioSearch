import requests
import json
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LENS_API_KEY = ""
API_URL = "https://api.lens.org/patent/search"

BIO_IPC_PATTERNS = ["C12*", "C07K*", "A61K*", "A01H*", "C12Q*", "C07H*", "A23L*"]

CUSTOM_IDA_KEYS = [
    "ATCC", "ECACC", "DSMZ", "NRRL", "IPOD", "CCTCC", "KCTC", "MTCC",
    "CGMCC", "GDMCC", "KCCM", "KCLRF", "KACC", "CNCM", "CBS", 
    "Westerdijk", "BCCM", "LMG", "IHEM", "MUCL", "IDAC", "NML", 
    "MCC", "NAIMCC", "VKM", "VKPM", "CECT", "BEA", "DBVPG", "IZSLER", 
    "CBA", "NMI", "PCM", "IAFB", "KPD", "NCIMB", "NCTC", "NCYC", 
    "CCAP", "IMI", "NIBSC", "NCMA", "IVS", "CChRGM", "CCM", "VTTCC", 
    "NCAIM", "MSCL", "CCMM", "CM-CNRG", "MUM", "UCCCB", "CCY", 
    "CCOS", "NBIMCC"
]
DEPOSIT_KEYWORDS = ["Budapest Treaty", "International Depository Authority", "biological deposit", "culture collection"] + CUSTOM_IDA_KEYS

def get_lens_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

def get_bio_patent_count():
    EXPIRED_STATUSES = ["EXPIRED", "LAPSED", "REVOKED", "CEASED"]
    
    query_payload = {
        "query": {
            "bool": {
                "must": [
                    {"terms": {"legal_status.patent_status": EXPIRED_STATUSES}},
                
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
        "size": 0,
        "include": ["total"]
    }
    
    headers = {"Authorization": f"Bearer {LENS_API_KEY}", "Content-Type": "application/json"}
    
    print(f"[*] Querying Lens API with EXTENDED Schema...")
    print(f"    - IPC Field: class_ipc.symbol")
    print(f"    - Patterns: {BIO_IPC_PATTERNS}")
    
    try:
        session = get_lens_session()
        response = session.post(API_URL, json=query_payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"[!] API Error: {response.status_code} - {response.text}")
            return 0
            
        data = response.json()
        
        if 'total' in data:
            return data['total']
        elif 'data' in data:
            return data.get('data', {}).get('total', 0)
        return 0
            
    except Exception as e:
        print(f"[ERROR] {e}")
        return 0

if __name__ == "__main__":
    count = get_bio_patent_count()
    
    if count > 0:
        print("\n========================================================")
        print(f"Total 'Bio-Liberated' Patents Found: {count:,}")
        print("========================================================")
        print(f"Criteria: Expired + Extended Bio-IPC + Deposit Keywords")
    else:
        print("\n[RESULT] No matching patents found.")
