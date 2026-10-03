import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from rag_engine import InsightEngine


class RecordingChatModel(BaseChatModel):
    """Returns canned replies in order and records every prompt it receives."""

    replies: list[str]
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        reply = self.replies[len(self.calls) - 1]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


def write_pdf(path, pages):
    """Writes a minimal PDF with one line of Helvetica text per page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = "%PDF-1.4\n"
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))
    return path


@pytest.fixture
def make_engine(tmp_path):
    def _make(replies=(), collection="test-collection"):
        return InsightEngine(
            collection_name=collection,
            persist_directory=str(tmp_path / "chroma_db"),
            llm=RecordingChatModel(replies=list(replies)),
            embeddings=DeterministicFakeEmbedding(size=32),
        )
    return _make


@pytest.fixture
def handbook_pdf(tmp_path):
    # Named like Streamlit's temp files, to prove citations use the uploaded name instead.
    return write_pdf(tmp_path / "tmpab12cd.pdf", ["Refunds take five business days.", "Card fees are two percent."])
