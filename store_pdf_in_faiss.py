import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a sentence embedding model (converts text to numbers)
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text() + "\n"
    return text

def store_text_in_faiss(texts):
    """Convert extracted text into embeddings and store in FAISS database."""
    embeddings = model.encode(texts)  # Convert text into numerical vectors
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)  # Create FAISS index
    index.add(np.array(embeddings))  # Add embeddings to FAISS

    faiss.write_index(index, "pdf_index.faiss")  # Save database
    print("PDF stored in FAISS successfully!")

# Extract text from PDF and store in FAISS
pdf_file = "sample.pdf"  # Replace with your PDF file name
extracted_text = extract_text_from_pdf(pdf_file)
store_text_in_faiss([extracted_text])  # Store in vector database
