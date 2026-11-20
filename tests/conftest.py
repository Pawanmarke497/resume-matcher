import io
import shutil
from pathlib import Path

import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.config import ROOT, Settings
from app.main import create_app

AI_RESUME = """Priya Sharma
AI Engineer
priya.sharma@example.com | +91 98765 43210

Summary
AI engineer with 4 years of experience building LLM applications.

Experience
AI Engineer, Acme Corp, 2022 - Present
- Built retrieval-augmented generation (RAG) pipelines with LangChain and LangGraph
- Served models through FastAPI with Docker on AWS
- Managed a Chroma vector store and evaluated embeddings; wrote prompt engineering guides

Software Engineer, Beta Ltd, 2020 - 2022
- Python backend services with PostgreSQL and pytest

Skills: Python, FastAPI, LangChain, LangGraph, RAG, LLMs, Docker, AWS, PostgreSQL, Git
"""


@pytest.fixture
def ai_resume_text() -> str:
    return AI_RESUME


@pytest.fixture
def client(tmp_path: Path):
    jobs_file = tmp_path / "jobs.json"
    shutil.copy(ROOT / "data" / "jobs.json", jobs_file)
    settings = Settings(llm_provider="offline", embedding_provider="offline", jobs_file=jobs_file)
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def docx_bytes(ai_resume_text: str) -> bytes:
    doc = Document()
    for line in ai_resume_text.splitlines():
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
