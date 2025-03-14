import streamlit as st
import google.generativeai as genai
import faiss
import pickle
import os
import json
import asyncio
import aiohttp
from PyPDF2 import PdfReader

# -------------------- Gemini AI Setup --------------------
try:
    GENAI_API_KEY = st.secrets["gemini_api_key"]
except KeyError:
    st.error("❌ Gemini API key not found in secrets.toml. Please add it and restart the app.")
    st.stop()

GENAI_MODEL = "gemini-2.0-flash"
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel(GENAI_MODEL)

# File paths for FAISS and text data storage
FAISS_INDEX_PATH = "faiss_index.pkl"
TEXT_DATA_PATH = "text_data.pkl"

# Check if FAISS index and text data exist, otherwise initialize
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TEXT_DATA_PATH):
    with open(FAISS_INDEX_PATH, "rb") as f:
        faiss_index = pickle.load(f)
    with open(TEXT_DATA_PATH, "rb") as f:
        text_data = pickle.load(f)
else:
    faiss_index = faiss.IndexFlatL2(768)  # Assuming embedding size is 768
    text_data = []  # Initialize text data storage
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(faiss_index, f)
    with open(TEXT_DATA_PATH, "wb") as f:
        pickle.dump(text_data, f)

# Streamlit UI
st.title("📜 PDF Chatbot with Gemini AI 🤖")
st.sidebar.header("Upload PDFs")

uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        text_data.append(text)

    # Save extracted text data
    with open(TEXT_DATA_PATH, "wb") as f:
        pickle.dump(text_data, f)

    st.sidebar.success("📂 PDFs uploaded successfully!")

# Function to search text in FAISS (dummy implementation for now)
def search_in_faiss(query):
    if text_data:
        return text_data[0]  # Returning first PDF text as dummy search result
    return "No relevant data found."

# Generate AI Answer
async def generate_answer(query, context):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GENAI_MODEL}:generateContent?key={GENAI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": f"Context: {context}\nQuestion: {query}"}]}]}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            result = await response.json()
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except KeyError:
                return "⚠️ Error: No valid response received from the AI model."

# Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("💬 Ask me anything about the uploaded PDFs:")
if query:
    context = search_in_faiss(query)
    answer = asyncio.run(generate_answer(query, context))
    
    # Store and display chat history
    st.session_state.chat_history.append((f"🧐 Q: {query}", f"🤖 A: {answer}"))
    
    for q, a in st.session_state.chat_history:
        st.write(q)
        st.write(a)
