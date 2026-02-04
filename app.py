import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- 1. SETUP PAGE CONFIG ---
st.set_page_config(page_title="DeepSearch Engine", page_icon="🔍", layout="wide")

# --- 2. LOAD RESOURCES (Cached for Speed) ---
@st.cache_resource
def load_resources():
    print("Loading resources...")
    # Load Corpus
    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    
    # Load Index
    index = faiss.read_index("semantic_search.index")
    
    # Load Models
    bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    return corpus, index, bi_encoder, cross_encoder

corpus, index, bi_encoder, cross_encoder = load_resources()

# --- 3. SEARCH FUNCTIONS ---
def dense_search(query, k=50):
    q_vec = bi_encoder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    _, idxs = index.search(q_vec, k)
    return [corpus[i] for i in idxs[0]]

def rerank_search(query, top_k=10):
    # 1. Retrieve Candidates (Top 50)
    candidates = dense_search(query, k=50)
    
    # 2. Re-Rank
    pairs = [[query, doc] for doc in candidates]
    scores = cross_encoder.predict(pairs)
    
    # 3. Sort
    results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return results[:top_k]

# --- 4. THE UI ---
st.title("🔍 Deep Semantic Search")
st.markdown("""
This system uses a **Bi-Encoder** for fast retrieval and a **Cross-Encoder** for high-precision re-ranking.
""")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Results to Show", 1, 20, 5)
    use_reranker = st.checkbox("Enable Re-Ranker", value=True)
    st.info("Re-ranker is slower (~200ms) but much more accurate.")

# Search Bar
query = st.text_input("Enter your search query:", placeholder="e.g., symptoms of flu")

if query:
    with st.spinner("Searching database..."):
        if use_reranker:
            results = rerank_search(query, top_k=top_k)
        else:
            # Just dense search, formatted to match rerank output
            docs = dense_search(query, k=top_k)
            results = [(d, 0.0) for d in docs] # Dummy scores for display

    st.success(f"Found {len(results)} relevant results")
    
    for rank, (text, score) in enumerate(results):
        with st.container():
            col1, col2 = st.columns([1, 15])
            with col1:
                st.markdown(f"### #{rank+1}")
            with col2:
                if use_reranker:
                    st.caption(f"Relevance Score: **{score:.4f}**")
                st.markdown(f"**{text}**")
            st.divider()
