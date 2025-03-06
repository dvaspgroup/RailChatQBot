import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF for PDF processing
import faiss
import os
import pickle
import time
import datetime
from sentence_transformers import SentenceTransformer

# Set up Gemini API
GENAI_API_KEY = "AIzaSyClBarKsOMvp-cll30MSoD-43IKnbyRHh4"  # Replace with your actual key
GENAI_MODEL = "gemini-1.5-pro"

genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel(GENAI_MODEL)

# Initialize SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Storage directories
UPLOAD_DIR = "uploaded_pdfs"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load or create FAISS index
if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
    faiss_index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        pdf_metadata = pickle.load(f)
else:
    faiss_index = faiss.IndexFlatL2(384)
    pdf_metadata = {}

# -------------------- UI DESIGN --------------------
st.set_page_config(page_title="RaiLChatbot", layout="wide")

# Custom CSS for better UI
st.markdown(
    """
    <style>
    /* Sidebar Styling */
    .stSidebar {
        background-color: #f8f9fa;
        padding: 20px;
        border-right: 2px solid #ddd;
    }
    
    /* Chat History Scrollable */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
    }

    /* User Message */
    .user-message {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 75%;
        align-self: flex-end;
    }

    /* AI Message */
    .ai-message {
        background-color: #F0F0F0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 75%;
        align-self: flex-start;
    }

    /* Source Styling */
    .source-text {
        color: #888;
        font-size: 12px;
    }

    /* User Avatar */
    .user-avatar {
        content: url("https://cdn-icons-png.flaticon.com/512/4333/4333609.png");
        height: 35px;
        width: 35px;
    }

    /* AI Avatar */
    .ai-avatar {
        content: url("https://cdn-icons-png.flaticon.com/512/4712/4712035.png");
        height: 35px;
        width: 35px;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar - User Authentication
st.sidebar.title("🔑 User Authentication")
user_role = st.sidebar.selectbox("Select Role", ["User", "Admin"])
st.sidebar.markdown(f"**Current Role:** {user_role}")

# Admin can upload PDFs
if user_role == "Admin":
    st.sidebar.subheader("📂 Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Drag and drop files here", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            # Save the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text and embeddings
            text_data = []
            pdf_doc = fitz.open(file_path)
            for page in pdf_doc:
                text_data.append(page.get_text("text"))
            pdf_text = "\n".join(text_data)

            # Generate embeddings
            text_embeddings = embedding_model.encode(text_data)

            # Add to FAISS index
            for i, embedding in enumerate(text_embeddings):
                faiss_index.add(embedding.reshape(1, -1))
                pdf_metadata[len(pdf_metadata)] = {"file": uploaded_file.name, "text": text_data[i]}

            st.sidebar.success(f"✅ {uploaded_file.name} uploaded successfully!")

        # Save index and metadata
        faiss.write_index(faiss_index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(pdf_metadata, f)

# -------------------- Chatbot UI --------------------
st.title("📜 RaiLChatBot 🤖")
st.markdown("💬 **Ask me anything about the uploaded PDFs:**")

query = st.text_input("Type your question here...", key="query")

# Chat history storage
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Query Processing ----------------
if st.button("Ask"):
    if not faiss_index.is_trained or faiss_index.ntotal == 0:
        st.error("⚠️ No PDFs uploaded yet. Please upload a file first.")
    else:
        # Convert query to embedding
        query_embedding = embedding_model.encode([query])

        # Search in FAISS index
        D, I = faiss_index.search(query_embedding, k=3)  # Get top 3 matches

        retrieved_texts = []
        source_docs = set()

        for idx in I[0]:
            if idx != -1:  # Ignore invalid indices
                retrieved_texts.append(pdf_metadata[idx]["text"])
                source_docs.add(pdf_metadata[idx]["file"])

        # Generate AI response
        context = "\n".join(retrieved_texts)
        response = model.generate_content(query + "\n\nContext:\n" + context)
        answer = response.text.strip()

        # Save chat history
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append(
            {
                "time": timestamp,
                "user": query,
                "bot": answer,
                "sources": ", ".join(source_docs) if source_docs else "Unknown",
            }
        )

# ---------------- Display Chat History ----------------
st.subheader("📜 Chat History")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for chat in reversed(st.session_state.chat_history):
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img class="user-avatar" />
            <div class="user-message">
                <b>You:</b> {chat['user']} <br>
                <span class="source-text">🕒 {chat['time']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img class="ai-avatar" />
            <div class="ai-message">
                <b>🤖 AI:</b> {chat['bot']} <br>
                <span class="source-text">📄 Source: {chat['sources']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
