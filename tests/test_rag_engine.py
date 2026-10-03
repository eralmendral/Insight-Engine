import pytest
from langchain_core.documents import Document

from conftest import write_pdf
from rag_engine import source_label


def test_citations_use_uploaded_name_and_one_based_pages(make_engine, handbook_pdf):
    engine = make_engine()
    assert engine.ingest_document(str(handbook_pdf), source_name="handbook.pdf") == 2

    labels = sorted(source_label(doc) for doc in engine.vector_store.similarity_search("fees", k=2))
    assert labels == ["handbook.pdf, page 1", "handbook.pdf, page 2"]


def test_reingesting_same_pdf_does_not_duplicate_chunks(make_engine, handbook_pdf):
    engine = make_engine()
    engine.ingest_document(str(handbook_pdf), source_name="handbook.pdf")
    engine.ingest_document(str(handbook_pdf), source_name="handbook.pdf")

    assert len(engine.vector_store.get()["ids"]) == 2


def test_pdf_without_text_is_rejected(make_engine, tmp_path):
    blank = write_pdf(tmp_path / "scan.pdf", [""])

    with pytest.raises(ValueError, match="No extractable text"):
        make_engine().ingest_document(str(blank), source_name="scan.pdf")


def test_first_question_skips_rewrite_call(make_engine, handbook_pdf):
    engine = make_engine()
    engine.ingest_document(str(handbook_pdf), source_name="handbook.pdf")

    assert engine.retrieve_context("What are the card fees?", chat_history=[])
    assert engine.llm.calls == []


def test_follow_up_is_rewritten_before_search(make_engine, monkeypatch):
    engine = make_engine(replies=["What are the card fees for refunds?"])
    searched = []
    monkeypatch.setattr(engine.vector_store, "similarity_search", lambda query, k: searched.append(query) or [])

    engine.retrieve_context("And the fees?", chat_history=[("human", "How do refunds work?"), ("ai", "...")])

    assert searched == ["What are the card fees for refunds?"]


def test_answer_prompt_numbers_sources_and_sends_question_once(make_engine):
    engine = make_engine(replies=["Card fees are two percent [1]."])
    docs = [Document(page_content="Card fees are two percent.", metadata={"source": "handbook.pdf", "page": 1})]

    answer = "".join(engine.generate_answer(
        "What are the card fees?", docs, chat_history=[("human", "Hi"), ("ai", "Hello!")]
    ))

    assert answer == "Card fees are two percent [1]."
    prompt = engine.llm.calls[0]
    assert "[1] handbook.pdf, page 2\nCard fees are two percent." in prompt[0].content
    assert [message.content for message in prompt].count("What are the card fees?") == 1


def test_sessions_are_isolated_and_reset_only_clears_own(make_engine, handbook_pdf):
    alice = make_engine(collection="session-alice")
    bob = make_engine(collection="session-bob")

    alice.ingest_document(str(handbook_pdf), source_name="handbook.pdf")
    assert not alice.is_empty() and bob.is_empty()

    bob.ingest_document(str(handbook_pdf), source_name="handbook.pdf")
    alice.reset()
    assert alice.is_empty() and not bob.is_empty()
