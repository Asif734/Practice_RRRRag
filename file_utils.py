import io, docx, PyPDF2, pandas as pd

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return " ".join([p.extract_text() or "" for p in reader.pages])

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return " ".join([para.text for para in doc.paragraphs])

def extract_text_from_excel(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    return " ".join(df.astype(str).fillna("").values.flatten().tolist())

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += (chunk_size - overlap)
    return chunks
