import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up Google Gemini API key
genai.configure(api_key="AIzaSyAzqdYX8kgx7owizsIjHxOqMk3X76CukG4")  # Replace with your actual key

# Load the embedding model globally (to avoid UnboundLocalError)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index
index = faiss.read_index("pdf_index.faiss")

def search_and_answer(query):
    """Search FAISS for relevant content and generate an answer using Gemini AI."""

    # Convert query into an embedding
    query_embedding = embedding_model.encode([query])  # Fixed issue here

    # Search FAISS index (Find top 3 closest matches)
    D, I = index.search(np.array(query_embedding), 3)

    # Simulate retrieved texts (Replace with actual stored texts)
    retrieved_texts = ["This is a sample response from the document."]

    # Use Google Gemini AI to generate an answer
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")  # Fixed the Gemini model initialization
    response = gemini_model.generate_content(f"Based on this information: {retrieved_texts}, answer this question: {query}")

    return response.text

# Example: Ask a question
user_query = "What is written in the PDF?"
answer = search_and_answer(user_query)
print("Answer:", answer)
