import pandas as pd
from langchain_community.llms import Ollama
import json
import re
import os
import sys
import time

INPUT_FILE = "Step3_Output.csv"
OUTPUT_FILE = "Step4_Output.csv"
MODEL_NAME = "llama3"
BATCH_SIZE = 10 

def extract_json_from_text(text):
    """
    Robust JSON extractor. Hunts for the JSON object in chatty responses.
    """
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match: return match.group(1)
    
    match = re.search(r'```\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match: return match.group(1)

    try:
        start = text.find('{')
        if start == -1: return None
        
        count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{': count += 1
            elif char == '}': count -= 1
            
            if count == 0:
                return text[start:i+1]
    except:
        pass
        
    return None

def run_extraction():
    print(f"[*] Connecting to Local LLM ({MODEL_NAME})...")
    try:
        llm = Ollama(model=MODEL_NAME, temperature=0, keep_alive="3h")
        llm.invoke("Hi") 
        print("[*] Connection Successful.")
    except Exception as e:
        print(f"[!] Error: {e}")
        return

    if not os.path.exists(INPUT_FILE):
        print(f"[!] Input file {INPUT_FILE} not found.")
        return
        
    df_input = pd.read_csv(INPUT_FILE)
    print(f"[*] Loaded {len(df_input)} snippets.")

    processed_keys = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_FILE, on_bad_lines='skip')
            if 'Lens_ID' in df_existing.columns:
                processed_keys = set(df_existing['Lens_ID'].astype(str) + "_" + df_existing['Accession_ID'].astype(str))
            print(f"[*] Resuming: {len(processed_keys)} assets done.")
        except:
            print("[!] Output file corrupted. Starting fresh.")
    else:
        columns = [
            "Accession_ID", "Repository", "Lens_ID", "Title", 
            "Bio_Name", "Bio_Strain", "Bio_Category", "Bio_Application", 
            "LLM_Status", "Raw_Response"
        ]
        pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False)

    print(f"[*] Starting Refined Extraction...")
    
    batch_buffer = []
    
    for i, row in df_input.iterrows():
        unique_key = str(row['Lens_ID']) + "_" + str(row['Accession_ID'])
        if unique_key in processed_keys: continue
            
        snippet = str(row.get('Context_Snippet', ''))
        title = str(row.get('Title', ''))
        
        if len(snippet) < 20:
            result = {"Bio_Name": "Unknown", "LLM_Status": "Skipped (Empty)", "Raw_Response": ""}
        else:
            # Context-Rich Prompt
            prompt = f"""
            Analyze this biological patent.
            
            PATENT TITLE: "{title}"
            CONTEXT SNIPPET: "{snippet[:2500]}"
            
            Task: Identify the biological material deposited as "{row['Accession_ID']}".
            
            Return a JSON object with:
            1. "name": Scientific species name (e.g. Escherichia coli). Use the Title as a hint.
            2. "strain": Specific strain ID (e.g. K-12).
            3. "category": (Bacteria, Fungi, Mammalian Cell Line, Virus, Plasmid, Other).
            4. "application": Industrial use (e.g. Antibody production).
            
            If a field is not found, use "Unknown". JSON ONLY.
            """
            
            raw_response = ""
            try:
                # Attempt 1: Direct
                raw_response = llm.invoke(prompt)
                json_str = extract_json_from_text(raw_response)
                
                if not json_str:
                    # Attempt 2: Self-Correction
                    repair_prompt = f"Extract the JSON object from this text:\n{raw_response}"
                    json_str = extract_json_from_text(llm.invoke(repair_prompt))
                
                if json_str:
                    data = json.loads(json_str)
                    result = {
                        "Bio_Name": data.get("name", "Unknown"),
                        "Bio_Strain": data.get("strain", "Unknown"),
                        "Bio_Category": data.get("category", "Unknown"),
                        "Bio_Application": data.get("application", "Unknown"),
                        "LLM_Status": "Success",
                        "Raw_Response": "" 
                    }
                else:
                    raise ValueError("No JSON found")
                    
            except Exception as e:
                # Fallback: Save Raw
                result = {
                    "Bio_Name": "Parse Error", "Bio_Strain": "See Raw",
                    "Bio_Category": "See Raw", "Bio_Application": "See Raw",
                    "LLM_Status": "JSON Failed",
                    "Raw_Response": raw_response.replace("\n", " ")[:500]
                }

        row_out = {
            "Accession_ID": row['Accession_ID'],
            "Repository": row['Repository'],
            "Lens_ID": row['Lens_ID'],
            "Title": row['Title'],
            **result
        }
        batch_buffer.append(row_out)
        
        if len(batch_buffer) >= BATCH_SIZE:
            pd.DataFrame(batch_buffer).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            status = result['Bio_Name'] if result['Bio_Name'] != "Unknown" else result['LLM_Status']
            print(f"\r\033[K -> Processed {len(processed_keys) + len(batch_buffer)}/{len(df_input)} | Last: {status}", end='')
            processed_keys.add(unique_key)
            batch_buffer = []
            sys.stdout.flush()

    if batch_buffer:
        pd.DataFrame(batch_buffer).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        
    print(f"\n[SUCCESS] Extraction Complete. File: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_extraction()
