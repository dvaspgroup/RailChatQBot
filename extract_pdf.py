import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text() + "\n"
    return text

# Test with a sample PDF
pdf_file = "sample.pdf"  # Replace with your PDF file name
extracted_text = extract_text_from_pdf(pdf_file)
print(extracted_text)  # Print extracted text
