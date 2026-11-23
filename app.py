import os
import streamlit as st
from dotenv import load_dotenv
from bot_brain import (
    DEFAULT_KNOWLEDGE,
    split_into_chunks,
    get_embeddings,
    extract_chunks_from_pdf,
    ask_bot,
)

load_dotenv()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# Keep sidebar collapsed by default so customers see a clean, focused chat screen
st.set_page_config(
    page_title="Client Support Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    /* Keep the tiny top-left sidebar toggle visible for admin access, hide Deploy button */
    .stAppDeployButton {display: none;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 740px;}
    .header-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .header-title {
        font-size: 1.35rem;
        font-weight: 600;
        color: #0F172A;
        margin: 0 0 0.25rem 0;
    }
    .header-sub {
        font-size: 0.9rem;
        color: #64748B;
        margin: 0;
    }
    .badge {
        display: inline-block;
        background: #F0FDFA;
        color: #0F766E;
        border: 1px solid #CCFBF1;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        margin-top: 0.65rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحباً بك! كيف يمكنني مساعدتك اليوم؟\n\nWelcome! How can I assist you today?"}
    ]

if "knowledge_chunks" not in st.session_state:
    default_chunks = split_into_chunks(DEFAULT_KNOWLEDGE)
    st.session_state.knowledge_chunks = default_chunks
    st.session_state.chunk_embeddings = get_embeddings(default_chunks)
    st.session_state.doc_label = "Al-Noor Dental Clinic Guide"

# Discreet Admin Sidebar (Collapsed by default)
with st.sidebar:
    st.markdown("### 🔐 Staff & Admin Access")
    entered_pass = st.text_input("Admin Password", type="password", placeholder="Demo password: admin123")

    if entered_pass == ADMIN_PASSWORD:
        st.success("Admin mode unlocked")
        st.caption("Upload an updated PDF document to refresh the assistant's knowledge.")

        uploaded_pdf = st.file_uploader("Upload Business PDF", type=["pdf"])

        if uploaded_pdf is not None and st.session_state.get("last_file") != uploaded_pdf.name:
            with st.spinner("Reading and learning your document..."):
                chunks, page_count = extract_chunks_from_pdf(uploaded_pdf)
                if chunks:
                    st.session_state.knowledge_chunks = chunks
                    st.session_state.chunk_embeddings = get_embeddings(chunks)
                    clean_name = uploaded_pdf.name.rsplit(".", 1)[0]
                    st.session_state.doc_label = f"{clean_name} ({page_count} pages)"
                    st.session_state.last_file = uploaded_pdf.name
                    st.success("Document updated successfully!")
                else:
                    st.error("We couldn't read text from this file. Please upload a text-based PDF.")
    elif entered_pass:
        st.error("Incorrect password.")
    else:
        st.caption("For business administrators only.")

    st.divider()
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "مرحباً بك! كيف يمكنني مساعدتك اليوم؟\n\nWelcome! How can I assist you today?"}
        ]
        st.rerun()

# Chic Header Card
st.markdown(f"""
<div class="header-card">
    <div class="header-title">💬 Smart Client Assistant</div>
    <div class="header-sub">Instant bilingual support in Arabic & English.</div>
    <div class="badge">● Knowledge Source: {st.session_state.doc_label}</div>
</div>
""", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    avatar = ":material/support_agent:" if msg["role"] == "assistant" else ":material/person:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat Input
if user_prompt := st.chat_input("Ask a question in Arabic, Darija, or English..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=":material/support_agent:"):
        with st.spinner("Checking our information..."):
            reply = ask_bot(
                st.session_state.messages,
                st.session_state.knowledge_chunks,
                st.session_state.chunk_embeddings,
            )
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})