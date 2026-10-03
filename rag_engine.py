import hashlib
import logging
import os
from typing import Iterator, List, Optional, Tuple

# Loaders & Splitters
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Models, Embeddings & Vector Store
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Schema & Interfaces
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from dotenv import load_dotenv
load_dotenv()

# Configure logging for production observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# (role, content) pairs, e.g. ("human", "What does it cost?"), oldest first.
ChatHistory = List[Tuple[str, str]]

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference context "
     "in the chat history, formulate a standalone question which can be understood "
     "without the chat history. Do NOT answer the question, just reformulate it if "
     "needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for question-answering tasks. Answer ONLY from the numbered "
     "sources below. Cite the sources you use inline, like [1] or [2]. If the sources "
     "don't contain the answer, say that you don't know.\n\n"
     "Sources:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def source_label(doc: Document) -> str:
    """Human-readable citation, e.g. 'handbook.pdf, page 3'."""
    name = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page")
    # PyPDF numbers pages from 0; readers count from 1.
    return f"{name}, page {page + 1}" if isinstance(page, int) else name


def format_sources(docs: List[Document]) -> str:
    """Numbers each chunk so the model's [n] citations match the UI's source list."""
    return "\n\n".join(
        f"[{i}] {source_label(doc)}\n{doc.page_content}" for i, doc in enumerate(docs, start=1)
    )


class InsightEngine:
    def __init__(
        self,
        collection_name: str = "insight_engine_core",
        persist_directory: str = "./chroma_db",
        llm: Optional[BaseChatModel] = None,
        embeddings: Optional[Embeddings] = None,
        k: int = 4,
    ):
        # Models are injectable so tests can run without an API key.
        self.llm = llm or ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"), temperature=0)
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
        self.k = k
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    def ingest_document(self, file_path: str, source_name: str) -> int:
        """
        Production pipeline: Load -> Split -> Embed -> Store.
        Returns the number of chunks stored.
        """
        try:
            logger.info(f"Starting ingestion for: {source_name}")

            # 1. Document Loading
            pages = PyPDFLoader(file_path).load()
            for page in pages:
                # PyPDFLoader records the temp-file path; cite the name the user uploaded.
                page.metadata["source"] = source_name

            # 2. Text Splitting (keeps PyPDF's page metadata on every chunk)
            chunks = self.text_splitter.split_documents(pages)
            if not chunks:
                raise ValueError("No extractable text found. Is this a scanned PDF?")

            # 3. Embedding & Storage
            # IDs derive from the file's content, so re-uploading the same PDF
            # overwrites its chunks instead of duplicating them.
            with open(file_path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()[:16]
            self.vector_store.add_documents(chunks, ids=[f"{digest}-{i}" for i in range(len(chunks))])

            logger.info(f"Successfully ingested {len(chunks)} chunks from {source_name}")
            return len(chunks)
        except Exception:
            logger.exception(f"Failed to ingest document: {source_name}")
            raise

    def is_empty(self) -> bool:
        return not self.vector_store.get(limit=1)["ids"]

    def reset(self) -> None:
        """Deletes every document in this engine's collection."""
        self.vector_store.reset_collection()

    def retrieve_context(self, query: str, chat_history: ChatHistory) -> List[Document]:
        """
        Retrieves relevant documents using the history-aware logic.
        Follow-ups like "How much does it cost?" are first rewritten into a standalone
        question, because the raw follow-up alone embeds poorly.
        """
        search_query = query
        if chat_history:
            search_query = (CONTEXTUALIZE_PROMPT | self.llm | StrOutputParser()).invoke(
                {"input": query, "chat_history": chat_history}
            )
            logger.info(f"Rewrote query {query!r} -> {search_query!r}")

        return self.vector_store.similarity_search(search_query, k=self.k)

    def generate_answer(self, query: str, context: List[Document], chat_history: ChatHistory) -> Iterator[str]:
        """Streams an answer grounded in the PRE-RETRIEVED context."""
        chain = QA_PROMPT | self.llm | StrOutputParser()
        yield from chain.stream({
            "context": format_sources(context),
            "input": query,
            "chat_history": chat_history,
        })
