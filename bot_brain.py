import os
import re
import time
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq
from google import genai
from google.genai import types

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
gemini_client = genai.Client()

DEFAULT_KNOWLEDGE = """
Business Name: Al-Noor Dental Clinic (عيادة النور لطب الأسنان)
Working Hours: Saturday to Thursday, 10:00 AM to 8:00 PM. Closed on Fridays.
Location: 45 Tahrir Street, Floor 3.
Services & Prices:
- Routine Checkup & Consultation: 200 EGP
- Teeth Cleaning & Polishing: 600 EGP
- Teeth Whitening: 2500 EGP (takes about 45 minutes in a single session)
Booking Policy: Patients must book at least 24 hours in advance via WhatsApp at +201000000000.
"""

def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Splits text along paragraph/sentence boundaries so sentences never cut off mid-word."""
    # Split into paragraphs/sentences first
    units = re.split(r'(?<=[.!?؟\n])\s+', text.strip())
    
    chunks = []
    current_chunk = ""

    for unit in units:
        # If adding the next sentence stays within chunk_size, append it
        if len(current_chunk) + len(unit) + 1 <= chunk_size:
            current_chunk = f"{current_chunk} {unit}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Keep the tail of the previous chunk for context overlap
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            # Snap overlap to the nearest space so we don't start mid-word
            if " " in overlap_text:
                overlap_text = overlap_text.split(" ", 1)[1]
            current_chunk = f"{overlap_text} {unit}".strip()

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def get_embeddings(texts: list[str]) -> np.ndarray | None:
    """Converts a list of text strings into a 2D NumPy array of semantic vectors."""
    for embed_model in ["gemini-embedding-001", "gemini-embedding-2"]:
        try:
            response = gemini_client.models.embed_content(
                model=embed_model,
                contents=texts,
            )
            vectors = [item.values for item in response.embeddings]
            return np.array(vectors, dtype=np.float32)
        except Exception as e:
            print(f"Embedding model {embed_model} skipped: {e}")
            continue
    return None

def extract_chunks_from_pdf(uploaded_file) -> tuple[list[str], int]:
    """Reads all pages of a PDF and returns (list_of_chunks, page_count)."""
    reader = PdfReader(uploaded_file)
    pages_text = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages_text.append(f"[Page {i}]\n{text}")
    full_text = "\n\n".join(pages_text)
    if not full_text.strip():
        return [], len(reader.pages)
    return split_into_chunks(full_text), len(reader.pages)

def keyword_fallback(query: str, chunks: list[str], top_k: int = 4) -> str:
    """Backup keyword search if embedding API is temporarily unreachable."""
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = [
        (sum(1 for w in query_words if len(w) > 2 and w in c.lower()), idx, c)
        for idx, c in enumerate(chunks)
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return "\n\n---\n\n".join(item[2] for item in scored[:top_k])

def retrieve_relevant_chunks(
    query: str,
    chunks: list[str],
    chunk_embeddings: np.ndarray | None,
    top_k: int = 4
) -> str:
    """Finds the top_k most semantically similar chunks using Cosine Similarity."""
    if len(chunks) <= top_k:
        return "\n\n---\n\n".join(chunks)

    # If we don't have precomputed vectors, fall back to keyword search
    if chunk_embeddings is None:
        return keyword_fallback(query, chunks, top_k)

    # Embed the user's question into a single vector
    query_vec_array = get_embeddings([query])
    if query_vec_array is None:
        return keyword_fallback(query, chunks, top_k)

    query_vec = query_vec_array[0]

    # Cosine Similarity formula: (A · B) / (||A|| * ||B||)
    dot_products = np.dot(chunk_embeddings, query_vec)
    norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_vec)
    similarities = dot_products / np.maximum(norms, 1e-10)

    # Get indices of the top_k highest similarity scores
    top_indices = np.argsort(similarities)[::-1][:top_k]
    best_chunks = [chunks[i] for i in top_indices]
    return "\n\n---\n\n".join(best_chunks)

def ask_bot(
    chat_history: list,
    knowledge_chunks: list[str],
    chunk_embeddings: np.ndarray | None
) -> str:
    """Retrieves semantically relevant sections and answers via Groq (with Gemini backup)."""
    latest_question = chat_history[-1]["content"]
    relevant_context = retrieve_relevant_chunks(
        latest_question, knowledge_chunks, chunk_embeddings, top_k=4
    )

    system_rules = f"""
    You are a polite, concise, and helpful customer support assistant.
    Answer the customer's question using ONLY the retrieved business document excerpts below.
    If the answer is not in the provided excerpts, politely state that you do not have that detail and suggest contacting human support.
    Reply in the exact same language (Arabic or English) that the customer used.

    --- RELEVANT BUSINESS EXCERPTS ---
    {relevant_context}
    """

    groq_messages = [{"role": "system", "content": system_rules}]
    for msg in chat_history[-6:]:
        groq_messages.append({"role": msg["role"], "content": msg["content"]})

    for groq_model in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]:
        try:
            completion = groq_client.chat.completions.create(
                model=groq_model,
                messages=groq_messages,
                temperature=0.2,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Groq {groq_model} skipped: {e}")
            continue

    formatted_contents = [
        types.Content(
            role="user" if msg["role"] == "user" else "model",
            parts=[types.Part.from_text(text=msg["content"])]
        )
        for msg in chat_history[-6:]
    ]

    for gemini_model in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
        try:
            response = gemini_client.models.generate_content(
                model=gemini_model,
                contents=formatted_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_rules,
                    temperature=0.2,
                )
            )
            return response.text
        except Exception:
            time.sleep(1)
            continue

    return "Sorry, our AI assistant is temporarily busy. Please try again in a moment."