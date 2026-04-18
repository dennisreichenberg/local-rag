"""Unit tests for document loading and chunking logic."""

import tempfile
from pathlib import Path

import pytest

from local_rag.loader import LoadError, chunk_text, load_document, load_text


def test_load_text_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write("Hello world. This is a test document.")
        tmp = Path(f.name)
    try:
        text = load_text(tmp)
        assert "Hello world" in text
    finally:
        tmp.unlink()


def test_load_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        tmp = Path(f.name)
    try:
        with pytest.raises(LoadError, match="Unsupported file type"):
            load_document(tmp)
    finally:
        tmp.unlink()


def test_chunk_text_basic():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 100


def test_chunk_text_overlap():
    text = " ".join(str(i) for i in range(200))
    chunks = chunk_text(text, size=50, overlap=10)
    # Verify overlap: last words of chunk N appear at start of chunk N+1
    words_end = chunks[0].split()[-10:]
    words_start = chunks[1].split()[:10]
    assert words_end == words_start


def test_chunk_empty_text():
    assert chunk_text("") == []


def test_chunk_text_short():
    text = "just a few words"
    chunks = chunk_text(text, size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text
