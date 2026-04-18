"""Document loaders: plain text, PDF, DOCX, Markdown."""

from __future__ import annotations

from pathlib import Path


class LoadError(Exception):
    pass


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise LoadError(str(e)) from e


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise LoadError("pypdf not installed — run: pip install pypdf") from e

    try:
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise LoadError(f"PDF read error: {e}") from e


def load_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as e:
        raise LoadError("python-docx not installed — run: pip install python-docx") from e

    try:
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        raise LoadError(f"DOCX read error: {e}") from e


_LOADERS: dict[str, object] = {
    ".txt": load_text,
    ".md": load_text,
    ".rst": load_text,
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise LoadError(f"Unsupported file type: {suffix}. Supported: {', '.join(_LOADERS)}")
    return loader(path)  # type: ignore[operator]


def chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += size - overlap
    return chunks
