import pandas as pd
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = "Step5_Output.csv"
INDEX_FILE = "bio_faiss.index"   # The Search Engine
META_FILE = "bio_meta.pkl"       # The Data Lookup
BATCH_SIZE = 100

def create_embeddings():
    print(f"[*] Loading Data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("[!] File not found. Run Step 3 first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    df = df[df['Bio_Name'].notna() & (df['Bio_Name'] != "Unknown")]
    print(f"[*] Found {len(df)} valid assets to embed.")

    print("[*] constructing Rich Documents...")
    documents = []
    metadata_lookup = []
    
    for idx, row in df.iterrows():
        bio_name = str(row.get('Bio_Name', 'Unknown'))
        bio_cat = str(row.get('Bio_Category', 'Unknown'))
        bio_app = str(row.get('Bio_Application', 'Unknown'))
        snippet = str(row.get('Context_Snippet', ''))
        title = str(row.get('Title', ''))
        
        rich_text = f"Organism: {bio_name}. Category: {bio_cat}. Application: {bio_app}. Title: {title}. Context: {snippet}"
        documents.append(rich_text)
        
        metadata_lookup.append({
            "accession_id": str(row['Accession_ID']),
            "repository": str(row['Repository']),
            "name": bio_name,
            "category": bio_cat,
            "application": bio_app,
            "title": title
        })

    print("[*] Loading Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("[*] Generating Vectors (This uses CPU/GPU)...")
    embeddings = model.encode(documents, batch_size=BATCH_SIZE, show_progress_bar=True)
    
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]

    print(f"[*] Building FAISS Index (Dim={dimension})...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"[*] Saving Index to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    
    print(f"[*] Saving Metadata to {META_FILE}...")
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata_lookup, f)
        
    print("\n[SUCCESS] Vector Database Built Successfully!")

def test_query():
    print("\n--- TEST QUERY: 'Bacteria that eats oil' ---")
    
    # Load
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta_data = pickle.load(f)
    model = SentenceTransformer('all-MiniLM-L6-v2')
  
    query = "Bacteria capable of degrading oil or hydrocarbons"
    vec = model.encode([query]).astype('float32')
    
    D, I = index.search(vec, k=3)
    
    for i, idx in enumerate(I[0]):
        if idx < len(meta_data):
            item = meta_data[idx]
            print(f"\nResult {i+1} (Dist: {D[0][i]:.2f}):")
            print(f"  {item['name']} ({item['category']})")
            print(f"  Repo: {item['repository']} {item['accession_id']}")
            print(f"  App: {item['application']}")

if __name__ == "__main__":
    create_embeddings()
    test_query()
