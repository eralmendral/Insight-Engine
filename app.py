import os
import tempfile
import uuid

import streamlit as st

from rag_engine import InsightEngine, source_label

# Page Config for a professional look
st.set_page_config(page_title="InsightEngine | AI Engineering Portfolio", layout="wide")

# Streamlit exports root-level secrets as env vars, so this covers both
# a local .env and Streamlit Cloud secrets.
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to `.env` (local) or the app's secrets (Streamlit Cloud).")
    st.stop()

# 1. Initialize the Engine in Session State
# This ensures we don't re-instantiate the model/db on every rerun.
# Each browser session gets its own collection, so visitors to the public demo
# never see (or wipe) each other's documents.
if "engine" not in st.session_state:
    st.session_state.engine = InsightEngine(collection_name=f"session-{uuid.uuid4().hex}")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

engine = st.session_state.engine


def render_sources(sources):
    """Shows the (label, excerpt) pairs that the answer's [n] citations point to."""
    with st.expander("📚 View Source Documents"):
        for i, (label, excerpt) in enumerate(sources, start=1):
            st.markdown(f"**[{i}]** {label}")
            st.caption(excerpt)


st.title("InsightEngine")
st.markdown("---")

# 2. Sidebar for Ingestion
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF Knowledge Base", type="pdf")

    if st.button("Ingest & Vectorize"):
        if not uploaded_file:
            st.warning("Choose a PDF first.")
        else:
            with st.spinner("Processing PDF..."):
                # Streamlit's UploadedFile is a BytesIO object.
                # PyPDFLoader requires a file path, so we use a temp file.
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                try:
                    chunk_count = engine.ingest_document(tmp.name, source_name=uploaded_file.name)
                    st.success(f"Added {chunk_count} chunks from {uploaded_file.name}.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                finally:
                    os.remove(tmp.name)

    st.markdown("---")
    if st.button("⚠️ Reset Knowledge Base"):
        engine.reset()
        st.session_state.chat_history = []
        st.success("Knowledge base cleared!")

# 3. Main Chat Interface
st.subheader("Query the Knowledge Base")

# Display conversation history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message.get("sources"):
            render_sources(message["sources"])
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a technical question about the uploaded docs..."):
    # Earlier turns only: the new question is passed to the chain separately as {input}.
    lc_history = [
        ("human" if msg["role"] == "user" else "ai", msg["content"])
        for msg in st.session_state.chat_history
    ]
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Retrieval & Generation
    with st.chat_message("assistant"):
        sources = []
        if engine.is_empty():
            # Skip the API calls entirely: there is nothing to retrieve from.
            full_response = "Upload a PDF and click **Ingest & Vectorize** first."
            st.markdown(full_response)
        else:
            with st.spinner("Analyzing documents..."):
                context_docs = engine.retrieve_context(prompt, lc_history)

            # Show sources before the answer streams, so claims can be checked as they appear.
            sources = [(source_label(doc), doc.page_content[:300] + "...") for doc in context_docs]
            render_sources(sources)
            full_response = st.write_stream(engine.generate_answer(prompt, context_docs, lc_history))

    # 5. Save to History (sources too, so earlier [n] citations stay resolvable)
    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "sources": sources})
