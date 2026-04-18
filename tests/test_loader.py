"""Unit tests for document loading and chunking logic."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_rag.loader import LoadError, chunk_text, load_document, load_text

# ---------------------------------------------------------------------------
# load_text
# ---------------------------------------------------------------------------


def test_load_text_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write("Hello world. This is a test document.")
        tmp = Path(f.name)
    try:
        text = load_text(tmp)
        assert "Hello world" in text
    finally:
        tmp.unlink()


def test_load_text_missing_file_raises_load_error():
    with pytest.raises(LoadError):
        load_text(Path("/nonexistent/path/does_not_exist.txt"))


def test_load_text_preserves_content(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Line one\nLine two\n", encoding="utf-8")
    assert load_text(f) == "Line one\nLine two\n"


# ---------------------------------------------------------------------------
# load_document routing
# ---------------------------------------------------------------------------


def test_load_document_txt_routing(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello txt")
    assert "hello txt" in load_document(f)


def test_load_document_md_routing(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# heading")
    assert "heading" in load_document(f)


def test_load_document_rst_routing(tmp_path):
    f = tmp_path / "doc.rst"
    f.write_text("Title\n=====")
    assert "Title" in load_document(f)


def test_load_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        tmp = Path(f.name)
    try:
        with pytest.raises(LoadError, match="Unsupported file type"):
            load_document(tmp)
    finally:
        tmp.unlink()


def test_load_document_unsupported_lists_valid_formats():
    tmp = Path("fake.csv")
    with pytest.raises(LoadError) as exc_info:
        load_document(tmp)
    assert ".txt" in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_pdf
# ---------------------------------------------------------------------------


def test_load_pdf_success():
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF page content"
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page]

    with patch("pypdf.PdfReader", return_value=mock_reader_instance):
        from local_rag.loader import load_pdf
        result = load_pdf(Path("fake.pdf"))

    assert "PDF page content" in result


def test_load_pdf_multiple_pages():
    pages = [MagicMock(), MagicMock()]
    pages[0].extract_text.return_value = "Page one"
    pages[1].extract_text.return_value = "Page two"
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = pages

    with patch("pypdf.PdfReader", return_value=mock_reader_instance):
        from local_rag.loader import load_pdf
        result = load_pdf(Path("fake.pdf"))

    assert "Page one" in result
    assert "Page two" in result


def test_load_pdf_page_with_no_text():
    mock_page = MagicMock()
    mock_page.extract_text.return_value = None
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page]

    with patch("pypdf.PdfReader", return_value=mock_reader_instance):
        from local_rag.loader import load_pdf
        result = load_pdf(Path("fake.pdf"))

    assert result == ""


def test_load_pdf_read_error_raises_load_error():
    with patch("pypdf.PdfReader", side_effect=Exception("corrupt PDF")):
        from local_rag.loader import load_pdf
        with pytest.raises(LoadError, match="PDF read error"):
            load_pdf(Path("bad.pdf"))


def test_load_pdf_missing_pypdf_raises_load_error():
    with patch.dict(sys.modules, {"pypdf": None}):
        from local_rag.loader import load_pdf
        with pytest.raises(LoadError, match="pypdf not installed"):
            load_pdf(Path("test.pdf"))


# ---------------------------------------------------------------------------
# load_docx
# ---------------------------------------------------------------------------


def test_load_docx_success():
    mock_para1 = MagicMock()
    mock_para1.text = "First paragraph"
    mock_para2 = MagicMock()
    mock_para2.text = "Second paragraph"
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para1, mock_para2]

    with patch("docx.Document", return_value=mock_doc):
        from local_rag.loader import load_docx
        result = load_docx(Path("fake.docx"))

    assert "First paragraph" in result
    assert "Second paragraph" in result


def test_load_docx_skips_empty_paragraphs():
    mock_para1 = MagicMock()
    mock_para1.text = "Real content"
    mock_para2 = MagicMock()
    mock_para2.text = "   "
    mock_doc = MagicMock()
    mock_doc.paragraphs = [mock_para1, mock_para2]

    with patch("docx.Document", return_value=mock_doc):
        from local_rag.loader import load_docx
        result = load_docx(Path("fake.docx"))

    assert "Real content" in result
    assert "   " not in result


def test_load_docx_read_error_raises_load_error():
    with patch("docx.Document", side_effect=Exception("corrupt docx")):
        from local_rag.loader import load_docx
        with pytest.raises(LoadError, match="DOCX read error"):
            load_docx(Path("bad.docx"))


def test_load_docx_missing_package_raises_load_error():
    with patch.dict(sys.modules, {"docx": None}):
        from local_rag.loader import load_docx
        with pytest.raises(LoadError, match="python-docx not installed"):
            load_docx(Path("test.docx"))


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_basic():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 100


def test_chunk_text_overlap():
    text = " ".join(str(i) for i in range(200))
    chunks = chunk_text(text, size=50, overlap=10)
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


def test_chunk_text_exact_size():
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, size=100, overlap=0)
    assert len(chunks) == 1
    assert len(chunks[0].split()) == 100


def test_chunk_text_no_overlap_produces_non_overlapping_chunks():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, size=50, overlap=0)
    assert len(chunks) == 2
    words_0 = set(chunks[0].split())
    words_1 = set(chunks[1].split())
    assert words_0.isdisjoint(words_1)


def test_chunk_text_whitespace_only():
    assert chunk_text("   ") == []
