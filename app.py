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

# DIAGNOSTICS: Check if files exist on Cloud
if not os.path.exists(INDEX_FILE):
    st.error(f"❌ Missing File: {INDEX_FILE}")
    st.write("Files in directory:", os.listdir('.'))
    st.stop()
if not os.path.exists(META_FILE):
    st.error(f"❌ Missing File: {META_FILE}")
    st.stop()

@st.cache_resource
def load_resources():
    # Load Index
    index = faiss.read_index(INDEX_FILE)
    # Load Metadata
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
    # Load Model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, metadata, model

# 2. LOAD WITH ERROR VISIBILITY
try:
    index, metadata, model = load_resources()
except Exception as e:
    st.error(f"⚠️ SYSTEM CRASH: {e}")
    st.info("Debugging Info:")
    st.json({
        "Python Version": os.sys.version,
        "Meta File Size (Bytes)": os.path.getsize(META_FILE),
        "Index File Size (Bytes)": os.path.getsize(INDEX_FILE)
    })
    st.stop()

# 3. HEADER
st.title("🧬 BioSearch")
st.markdown(f"Search **{len(metadata):,}** open-source organisms liberated from expired patents.")

# 4. SIDEBAR
st.sidebar.header("Filter Results")
# Handle potential missing categories
valid_cats = [m['category'] for m in metadata if isinstance(m, dict) and 'category' in m]
all_cats = sorted(list(set(valid_cats))) if valid_cats else []
cat_filter = st.sidebar.multiselect("Category", all_cats)

# 5. SEARCH ENGINE
query = st.text_input("What are you looking for?", placeholder="e.g. 'Yeast for ethanol' or 'CHO cell line'")

if query:
    with st.spinner("Scanning Bio-Archive..."):
        vec = model.encode([query]).astype('float32')
        D, I = index.search(vec, k=100)
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(metadata):
                item = metadata[idx]
                
                if cat_filter and item['category'] not in cat_filter: 
                    continue
                
                score = round((1 - dist) * 100, 1)
                results.append({**item, "score": score})
                if len(results) >= 15: break
        
        if results:
            st.success(f"Found {len(results)} matches.")
            for res in results:
                with st.expander(f"**{res['name']}** ({res['category']}) - {res['score']}% Match"):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**💡 Application:** {res.get('application', 'N/A')}")
                        st.markdown(f"**📜 Patent:** *{res.get('title', 'N/A')}*")
                        if res.get('lens_id'):
                            st.markdown(f"🔗 [**View on Lens.org**](https://www.lens.org/lens/patent/{res['lens_id']})")
                    with c2:
                        st.metric("Repository", res.get('repository', 'Unknown'))
                        st.code(res.get('accession_id', ''))
        else:
            st.warning("No matches found.")
else:
    st.info("👆 Enter a query above to start discovery.")
