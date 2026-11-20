"""Turn an uploaded resume file (PDF / DOCX / TXT) into plain text."""

from __future__ import annotations

import io
import re
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedFileType(ValueError):
    pass


class UnreadableFile(ValueError):
    pass


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(content: bytes) -> str:
    doc = Document(io.BytesIO(content))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def extract_text(filename: str, content: bytes) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type '{ext or 'unknown'}'. "
            f"Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    try:
        if ext == ".pdf":
            raw = _read_pdf(content)
        elif ext == ".docx":
            raw = _read_docx(content)
        else:
            raw = content.decode("utf-8", errors="ignore")
    except Exception as exc:  # corrupt / encrypted / not really that format
        raise UnreadableFile(f"Could not read the {ext} file: {exc}") from exc
    return _normalize(raw)
