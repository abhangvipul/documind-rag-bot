import os
# Suppress the Hugging Face hub warning cleanly
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
import faiss
import numpy as np

# 🌟 Premium tech dashboard layout configuration
st.set_page_config(page_title="DocuMind AI Ultra", page_icon="🔮", layout="centered")

# 🎨 Custom Modern Light Theme Glassmorphism Interface Styling
st.markdown("""
    <style>
    /* Clean professional light canvas background layout */
    .stApp {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        color: #1F2937;
    }
    
    /* Modern vibrant gradient heading design style */
    .main-title {
        font-size: 46px;
        font-weight: 900;
        background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: -1px;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 16px;
        text-align: center;
        color: #4B5563;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    /* Premium Frosted Glass Card layout panels - Light Mode */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }
    
    /* Elegant Clean Response Output Box layout elements */
    .response-container {
        background: linear-gradient(145deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #BFDBFE;
        padding: 24px;
        border-radius: 14px;
        color: #1E3A8A;
        font-size: 16px;
        line-height: 1.7;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
    }
    
    /* File Uploader browse button styling - High contrast bright corporate blue */
    [data-testid="stFileUploader"] button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #1D4ED8 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: 0.3s ease;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.3) !important;
    }
    [data-testid="stFileUploader"] button p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Submit Button custom gradient theme text styling */
    .stFormSubmitButton button {
        background: linear-gradient(135deg, #4F46E5 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.25) !important;
        transition: 0.3s ease !important;
        width: 100% !important;
    }
    .stFormSubmitButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.4) !important;
    }
    .stFormSubmitButton button p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar restyling parameters override block */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E7EB;
    }
    
    /* Make standard interactive text crisp inside light frames */
    p, label {
        color: #374151 !important;
    }
    h3, h4 {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    
    /* Dynamic fix for the file uploader drop text color visibility */
    [data-testid="stFileUploadDropzone"] div {
        color: #4B5563 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Application Tech Header
st.markdown('<div class="main-title">🔮 DocuMind AI Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Your intelligent document assistant. Upload a PDF and get answers from its contents.</div>', unsafe_allow_html=True)

# 🧠 Cache heavy neural architectures inside local thread memory pool
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return embedding_model, tokenizer, model

with st.spinner("⚡ Initializing Neural Network Engines... Please hold..."):
    embedding_model, tokenizer, model = load_models()

# Sidebar Panel custom configurations
st.sidebar.markdown("### 🛠️ Core Parameters")
chunk_size = st.sidebar.slider("Chunk Window Length", 200, 1000, 500, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap Window", 50, 300, 100, step=10)
st.sidebar.markdown("---")
st.sidebar.caption("🔒 Architecture Security: 100% Local Processing Network")

# File input area
st.markdown("### 📂 Ingest Document Workspace")
uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None:
    with st.spinner("🧬 Extracting contextual text layers from document mapping..."):
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            if page.extract_text():
                full_text += page.extract_text() + "\n"
        
        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunks.append(full_text[start:end].strip())
            start += (chunk_size - chunk_overlap)

    if chunks:
        # Styled modern data summary metrics card layout component
        st.markdown(f"""
            <div class="glass-card">
                <h4 style="margin:0 0 8px 0; color:#2563EB; font-size:16px; text-transform: uppercase; letter-spacing: 1px;">📊 Vector Analytics Stream</h4>
                <p style="margin:0; color:#4B5563; font-size:15px;">
                    File Pipeline Status: <span style="color:#10B981; font-weight:bold;">Active</span> &nbsp;|&nbsp; 
                    Mathematical Nodes Generated: <span style="color:#7C3AED; font-weight:bold;">{len(chunks)} Chunks</span>
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Generate FAISS vector database metrics directly into RAM
        with st.spinner("🧠 Mapping embeddings directly onto local vector plane matrix..."):
            doc_embeddings = embedding_model.encode(chunks)
            dimension = doc_embeddings.shape[1] 
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(doc_embeddings).astype('float32'))

        # Search Query Interaction Block
        st.markdown("### 💬 System Query Console")
        with st.form(key="modern_rag_form"):
            user_query = st.text_input("Pose a query to the embedded document knowledge matrix:", placeholder="What insight do you want to unlock from this file?", label_visibility="collapsed")
            submit_button = st.form_submit_button(label="🔍 Find Answer")

        if submit_button and user_query:
            with st.spinner("📡 Parsing vector semantic fields and compiling optimal context..."):
                query_embedding = embedding_model.encode([user_query])
                D, I = index.search(np.array(query_embedding).astype('float32'), k=2)
                
                retrieved_context = ""
                for idx in I[0]: 
                    if 0 <= idx < len(chunks):
                        retrieved_context += chunks[idx] + "\n"
                
                prompt = f"""Answer the question based strictly on the context provided below from the document. 
If the answer is not in the context, say "The requested information is not present in the document."

Context: 
{retrieved_context}

Question: {user_query}

Answer:"""
                
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(**inputs, max_new_tokens=150)
                final_answer = tokenizer.decode(outputs, skip_special_tokens=True)
            
            # Print the modernized layout display answer box blocks
            st.markdown("#### ✨ AI Generated Answer:")
            st.markdown(f'<div class="response-container">✨ {final_answer}</div>', unsafe_allow_html=True)
            
            # Diagnostic drop-down block
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🛠️ View Ground Truth Context Node Fragments"):
                st.write(retrieved_context if retrieved_context.strip() else "No related vector blocks fetched.")
    else:
        st.error("No extractable character streams discovered. The file formatting might be image-exclusive.")
