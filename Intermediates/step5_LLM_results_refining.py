import pandas as pd
import re
import json
import os

INPUT_FILE = "Step4_Output.csv"
OUTPUT_FILE = "Step5_Output.csv"

def aggressive_rescue(raw_text):
    """
    Rescues data from broken JSON in the 'Raw_Response' column.
    Handles messy LLM output like: "name": "E. coli" and "strain": "K12"
    """
    if not isinstance(raw_text, str) or len(raw_text) < 10: return None
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw_text, re.IGNORECASE)
    name = name_match.group(1) if name_match else "Unknown"
    
    cat_match = re.search(r'"category"\s*:\s*"([^"]+)"', raw_text, re.IGNORECASE)
    category = cat_match.group(1) if cat_match else "Unknown"
    
    app_match = re.search(r'"application"\s*:\s*"([^"]+)"', raw_text, re.IGNORECASE)
    application = app_match.group(1) if app_match else "Unknown"
    
    if name != "Unknown" or category != "Unknown":
        return {
            "Bio_Name": name, 
            "Bio_Category": category, 
            "Bio_Application": application, 
            "LLM_Status": "Rescued"
        }
    return None

def clean_generics(row):
    """
    Moves generic terms from 'Bio_Name' to 'Bio_Category'.
    Fixes: Human, Hybridoma, Plasmid, Bacteria, etc.
    """
    name = str(row['Bio_Name']).strip()
    category = str(row['Bio_Category']).strip()
    
    if name.lower() == "human":
        row['Bio_Name'] = "Unknown" 
        row['Bio_Category'] = "Human Cell Line"
        return row

    if "hybridoma" in name.lower():
        row['Bio_Name'] = "Unknown"
        row['Bio_Category'] = "Hybridoma"
        return row
        
    if "plasmid" in name.lower() or "vector" in name.lower():
        row['Bio_Name'] = "Unknown"
        row['Bio_Category'] = "Plasmid/Vector"
        return row

    generics = ["bacteria", "fungi", "yeast", "virus", "mammalian cell line", "cell line"]
    if name.lower() in generics:
        row['Bio_Name'] = "Unknown"
        row['Bio_Category'] = name.title() if category in ["Unknown", "Error", "Other"] else category
        return row

    return row

def run_polish():
    print(f"[*] Loading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"[!] File not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"[*] Polishing {len(df)} rows...")
    
    # TRACKING METRICS
    stats = {"Rescued": 0, "Cleaned_Human": 0, "Cleaned_Hybridoma": 0, "Cleaned_Plasmid": 0}
    
    # 1. RESCUE FAILED JSON
    for i, row in df.iterrows():
        if "Failed" in str(row['LLM_Status']) or "Error" in str(row['LLM_Status']):
            data = aggressive_rescue(row['Raw_Response'])
            if data:
                df.at[i, 'Bio_Name'] = data['Bio_Name']
                df.at[i, 'Bio_Category'] = data['Bio_Category']
                df.at[i, 'Bio_Application'] = data['Bio_Application']
                df.at[i, 'LLM_Status'] = "Rescued"
                stats["Rescued"] += 1

    def tracked_clean(row):
        old_name = str(row['Bio_Name']).lower()
        new_row = clean_generics(row)
        new_cat = str(new_row['Bio_Category'])
        
        if old_name == "human" and new_row['Bio_Name'] == "Unknown": stats["Cleaned_Human"] += 1
        elif "hybridoma" in old_name and new_row['Bio_Name'] == "Unknown": stats["Cleaned_Hybridoma"] += 1
        elif ("plasmid" in old_name or "vector" in old_name) and new_row['Bio_Name'] == "Unknown": stats["Cleaned_Plasmid"] += 1
        
        return new_row

    df = df.apply(tracked_clean, axis=1)
    
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print(f" [SUCCESS] POLISHING COMPLETE")
    print("="*40)
    print(f" -> Rescued Failed Rows:    {stats['Rescued']}")
    print(f" -> Fixed 'Human' entries:  {stats['Cleaned_Human']}")
    print(f" -> Fixed 'Hybridomas':     {stats['Cleaned_Hybridoma']}")
    print(f" -> Fixed 'Plasmids':       {stats['Cleaned_Plasmid']}")
    print("-" * 40)
    print(f" Final Dataset Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_polish()
