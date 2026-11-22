import os
import re
import time
from dotenv import load_dotenv
from pypdf import PdfReader 
from groq import Groq
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

# 1. Load your secret API key from the .env file
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# 2. Initialize the Gemini client (it automatically reads GEMINI_API_KEY)
gemini_client = genai.Client()

# 3. Sample business knowledge (we will replace this with real PDF uploads next!)
DEFAULT_KNOWLEDGE = """
Business Name: Al-Noor Dental Clinic (عيادة النور لطب الأسنان)
Working Hours: Saturday to Thursday, 10:00 AM to 8:00 PM. Closed on Fridays.
Location: 45 Tahrir Street, Floor 3.
Services & Prices:
- Routine Checkup & Consultation: 200 EGP
- Teeth Cleaning & Polishing: 600 EGP
- Teeth Whitening: 2500 EGP
Booking Policy: Patients must book at least 24 hours in advance via WhatsApp at +201000000000.
"""

def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Splits full document text into overlapping chunks without losing a single word."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_chunks_from_pdf(uploaded_file) -> tuple[list[str], int]:
    """Reads all pages of a PDF and returns (list_of_chunks, page_count)."""
    reader = PdfReader(uploaded_file)
    pages_text = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text :
            pages_text.append(f"[Page {i}]\n{text}")
    full_text = "\n\n".join(pages_text)
    if not full_text.strip():
        return [], len(reader.pages)
    return split_into_chunks(full_text), len(reader.pages)

def retrieve_relevant_chunks(query: str, chunks: list[str], top_k: int = 4) -> str:
    """Finds the most relevant chunks from the entire PDF for the user's question."""
    if len(chunks) <= top_k:
        return "\n\n---\n\n".join(chunks)

    # Extract meaningful words (supports both Arabic and English)
    query_words = set(re.findall(r"\w+", query.lower()))
    scored_chunks = []

    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(1 for word in query_words if len(word) > 2 and word in chunk_lower)
        scored_chunks.append((score, idx, chunk))

    # Sort by highest keyword match; if no keywords match (e.g., "summarize"), return opening chunks
    scored_chunks.sort(key=lambda x: (-x[0], x[1]))
    best_chunks = [item[2] for item in scored_chunks[:top_k]]
    return "\n\n---\n\n".join(best_chunks)


def ask_bot(chat_history: list, knowledge_chunks: list[str]) -> str:
    """
    Uses Groq for near-instant replies, with Gemini as an automatic backup.
    Sends the full conversation history + business knowledge
    so the bot remembers previous messages in the chat.
    """
    """Retrieves relevant sections from the full document and answers via Groq (with Gemini backup)."""
    latest_question = chat_history[-1]["content"]
    relevant_context = retrieve_relevant_chunks(latest_question, knowledge_chunks, top_k=4)

    # 4. System Instructions: The strict rules the AI must obey
    system_rules = f"""
    You are a polite, helpful customer support assistant for Al-Noor Dental Clinic.
    Answer the customer's question using ONLY the business information provided below.
    If the answer is NOT in the business information, politely say you don't have that information and suggest contacting WhatsApp support.
    Reply in the exact same language (Arabic or English) that the customer used.

    --- BUSINESS INFORMATION ---
    {relevant_context}
    """

    # 1. Try Groq first (ultra-fast)
    groq_messages = [{"role": "system", "content": system_rules}]
    for msg in chat_history[-6:]:  # Keep last 6 turns so conversation memory stays fast
        groq_messages.append({"role": msg["role"], "content": msg["content"]})

    for groq_model in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]:
        try:
            completion = groq_client.chat.completions.create(
                model=groq_model,
                messages=groq_messages, # type: ignore
                temperature=0.2,
            )
            return completion.choices[0].message.content # type: ignore
        except Exception as e:
            print(f"Groq {groq_model} skipped: {e}")
            continue

    # 2. Automatic fallback to Gemini if Groq hits a limit
    # Convert our simple chat history into Gemini's expected Content format
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
                    temperature=0.2,  # Low temperature = more factual, less creative guessing
                )
            )
            return response.text # type: ignore
        except (ServerError, ClientError) as e:
            print(f"Skipping {gemini_model} due to: {e}")
            time.sleep(1)
            continue

    return "Sorry, all AI providers are temporarily busy. Please try again in a moment."

    