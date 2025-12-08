import streamlit as st
import faiss
import pickle
import os
import glob
from sentence_transformers import SentenceTransformer

# 0. GITHUB FIX (File Stitching)
def reconstruct_file(filename):
    if not os.path.exists(filename):
        parts = sorted(glob.glob(f"{filename}.part*"))
        if parts:
            with open(filename, "wb") as outfile:
                for part in parts:
                    with open(part, "rb") as infile:
                        outfile.write(infile.read())

reconstruct_file("bio_faiss.index")
reconstruct_file("bio_meta.pkl")

# 1. SETUP
st.set_page_config(page_title="BioSearch", page_icon="🧬", layout="wide")
INDEX_FILE = "bio_faiss.index"
META_FILE = "bio_meta.pkl"

@st.cache_resource
def load_resources():
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return index, metadata, model
    except Exception as e:
        return None, None, None

index, metadata, model = load_resources()

# 2. HEADER
st.title("🧬 BioSearch")
st.markdown("### The Search Engine for Liberated Biology")
st.markdown(f"Search **{len(metadata) if metadata else 0:,}** open-source organisms, cell lines, and plasmids liberated from expired patents.")

if index is not None:
    # SIDEBAR
    st.sidebar.header("Filter Results")
    all_cats = sorted(list(set([m['category'] for m in metadata if m['category'] != "Unknown"])))
    cat_filter = st.sidebar.multiselect("Category", all_cats)
    
    # SEARCH
    query = st.text_input("What are you looking for?", placeholder="e.g. 'Yeast for ethanol' or 'CHO cell line'")
    
    if query:
        with st.spinner("Scanning Bio-Archive..."):
            vec = model.encode([query]).astype('float32')
            D, I = index.search(vec, k=50)
            
            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx < len(metadata):
                    item = metadata[idx]
                    
                    if cat_filter and item['category'] not in cat_filter: 
                        continue
                    
                    score = round((1 - dist) * 100, 1)
                    results.append({**item, "score": score})
                    if len(results) >= 10: break
            
            if results:
                st.success(f"Found {len(results)} matches.")
                
                for res in results:
                    with st.expander(f"**{res['name']}** ({res['category']}) - {res['score']}% Match"):
                        c1, c2 = st.columns([3, 1])
                        
                        with c1:
                            st.markdown(f"**💡 Application:** {res['application']}")
                            st.markdown(f"**📜 Source Patent:** *{res['title']}*")
                            
                            # LINK GENERATION
                            if res.get('lens_id'):
                                lens_url = f"https://www.lens.org/lens/patent/{res['lens_id']}"
                                st.markdown(f"🔗 [**View Original Patent on Lens.org**]({lens_url})")
                            else:
                                st.caption("No Lens ID available")
                        
                        with c2:
                            st.metric("Repository", res['repository'])
                            st.code(res['accession_id'])
                            
                            # Repo Links
                            repo_url = "#"
                            if res['repository'] == "ATCC":
                                repo_url = f"https://www.atcc.org/products/{res['accession_id']}"
                            elif res['repository'] == "DSMZ":
                                repo_url = f"https://www.dsmz.de/collection/catalogue/details/culture/DSM-{res['accession_id']}"
                            
                            if repo_url != "#":
                                st.markdown(f"[Order from {res['repository']}]({repo_url})")
            else:
                st.warning("No matches found.")
    
    else:
        st.info("👆 Enter a query above to start discovery.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Organisms", len(metadata))
        c2.metric("Categories", len(all_cats))
        c3.metric("Backend", "FAISS + MiniLM")

else:
    st.error("System Error: Could not load database.")
    st.info("Ensure bio_faiss.index and bio_meta.pkl are uploaded.")
