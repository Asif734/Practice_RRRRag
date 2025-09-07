from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
from file_utils import extract_text_from_pdf, extract_text_from_docx, extract_text_from_excel, chunk_text
from embeddings import embed_texts
from pinecone_client import upsert_vectors, query_index
from llm import ask_ollama
from models import QueryRequest, QueryResponse
import uuid
from redis_memory import set_cached_answer, get_cached_answer

app = FastAPI(title="Modular Multilingual RAG with Pinecone + Ollama")

# ---------------- Routes ----------------
@app.post("/ingest")
async def ingest(
    file: Optional[UploadFile] = File(None),
    text_input: Optional[str] = Form(None),
    source_id: Optional[str] = Form(None)
):
    if not file and not text_input:
        return {"status": "error", "message": "Either file or text_input must be provided."}

    # Get text from file if provided
    text = ""
    source_name = source_id or "user_input"
    if file:
        filename = file.filename.lower()
        content = await file.read()
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(content)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(content)
        elif filename.endswith((".xls", ".xlsx")):
            text = extract_text_from_excel(content)
        else:
            text = content.decode("utf-8")
        source_name = file.filename

    # Append text input if provided
    if text_input:
        text = text + "\n" + text_input if text else text_input

    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    # ✅ Use source_name instead of file.filename
    metadata = [{"source": source_name, "text": c} for c in chunks]

    upsert_vectors(ids, vectors, metadata)

    return {"status": "ok", "chunks_added": len(chunks)}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    session_id = req.session_id or "default"

    # check cache
    cached = get_cached_answer(req.query, session_id)
    if cached:
        return cached

    q_emb = embed_texts([req.query])[0]
    results = query_index(vector=q_emb, top_k=req.top_k, include_metadata=True)
    context = "\n\n".join([m["metadata"]["text"] for m in results["matches"]])

    prompt = f"""
You are a helpful, friendly, multilingual assistant.
Answer naturally and concisely.
Use the provided context if it is relevant.
If the user input is unrelated to the context, respond naturally as a human would.

Context:
{context if context else 'No context available.'}

User: {req.query}
Assistant:
"""

    answer = ask_ollama(prompt, session_id=session_id)
    sources = [{"id": m["id"], "score": m["score"], "source": m["metadata"]["source"]} for m in results["matches"]]

    # cache in Redis
    set_cached_answer(req.query, session_id, answer, sources)

    return {"answer": answer, "sources": sources}


