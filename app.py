import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURATION
# ==========================================
INDEX_FILE = "bio_faiss.index"
META_FILE = "bio_meta.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'

# ==========================================
# LOAD RESOURCES (Cached for Speed)
# ==========================================
@st.cache_resource
def load_resources():
    # 1. Load Vector Index
    index = faiss.read_index(INDEX_FILE)
    
    # 2. Load Metadata
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
        
    # 3. Load Model
    model = SentenceTransformer(MODEL_NAME)
    
    return index, metadata, model

try:
    index, metadata, model = load_resources()
    APP_READY = True
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.info("Did you run step4_faiss_embeddings.py first?")
    APP_READY = False

# ==========================================
# UI LAYOUT
# ==========================================
st.set_page_config(page_title="Bio-Broker", page_icon="🧬", layout="wide")

st.title("🧬 Bio-Broker: The Liberated Organism Search")
st.markdown("""
**Find expired, open-source biological assets for your research.**
This engine searches over **19,000+** organisms liberated from patent monopolies.
""")

# SIDEBAR FILTERS
st.sidebar.header("🔍 Filters")
category_filter = st.sidebar.multiselect(
    "Filter by Category",
    options=sorted(list(set([m['category'] for m in metadata]))),
    default=[]
)

# SEARCH BAR
query = st.text_input("Describe what you need:", placeholder="e.g., 'Bacteria capable of degrading oil' or 'Cell line for antibody production'")

if APP_READY and query:
    # 1. Vector Search
    with st.spinner("Searching the bio-archive..."):
        # Encode Query
        vec = model.encode([query]).astype('float32')
        
        # Search Index (Get top 50 to allow for filtering)
        D, I = index.search(vec, k=50)
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(metadata):
                item = metadata[idx]
                
                # Apply Category Filter
                if category_filter and item['category'] not in category_filter:
                    continue
                
                # Format Score
                score = round((1 - dist) * 100, 1) # Rough similarity score
                
                results.append({
                    "Score": score,
                    "Organism": item['name'],
                    "Category": item['category'],
                    "Application": item['application'],
                    "Repository": item['repository'],
                    "Accession ID": item['accession_id'],
                    "Patent Title": item['title']
                })
                
                if len(results) >= 10: # Only show top 10 after filtering
                    break
    
    # 2. Display Results
    if results:
        st.success(f"Found {len(results)} relevant assets.")
        
        for res in results:
            with st.expander(f"**{res['Organism']}** ({res['Category']}) - {res['Score']}% Match"):
                c1, c2 = st.columns([3, 1])
                
                with c1:
                    st.markdown(f"**💡 Application:** {res['Application']}")
                    st.markdown(f"**📜 Source Patent:** *{res['Patent Title']}*")
                
                with c2:
                    st.metric("Repository", res['Repository'])
                    st.code(res['Accession ID'])
                    
                    # Link to Repository (Heuristic)
                    repo_url = "#"
                    if res['Repository'] == "ATCC":
                        repo_url = f"https://www.atcc.org/products/{res['Accession ID']}"
                    elif res['Repository'] == "DSMZ":
                        repo_url = f"https://www.dsmz.de/collection/catalogue/details/culture/DSM-{res['Accession ID']}"
                    
                    if repo_url != "#":
                        st.markdown(f"[Order from {res['Repository']}]({repo_url})")
    else:
        st.warning("No matches found. Try broadening your search terms.")

elif APP_READY:
    st.info("👆 Enter a query above to start discovery.")
    
    # Show Stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Organisms", len(metadata))
    c2.metric("Categories", len(set([m['category'] for m in metadata])))
    c3.metric("Search Backend", "FAISS + MiniLM")
