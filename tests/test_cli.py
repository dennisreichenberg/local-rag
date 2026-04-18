"""Tests for CLI commands using Click's CliRunner."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from local_rag.cli import main
from local_rag.embedder import EmbedError
from local_rag.loader import LoadError


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _suppress_rich(monkeypatch):
    """Silence Rich console output so it doesn't bleed into test stdout."""
    with patch("local_rag.cli.console"), patch("local_rag.cli.err"):
        yield


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


def test_list_cmd_empty_store(runner):
    with patch("local_rag.store.list_sources", return_value=[]):
        result = runner.invoke(main, ["list"])
    assert result.exit_code == 0


def test_list_cmd_with_sources(runner):
    with patch("local_rag.store.list_sources", return_value=["a.txt", "b.txt"]):
        result = runner.invoke(main, ["list"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# add command
# ---------------------------------------------------------------------------


def test_add_cmd_load_error_continues_to_next_file(runner, tmp_path):
    doc = tmp_path / "bad.txt"
    doc.write_text("x")
    with patch("local_rag.loader.load_document", side_effect=LoadError("cannot read")):
        result = runner.invoke(main, ["add", str(doc)])
    assert result.exit_code == 0


def test_add_cmd_empty_text_skips_file(runner, tmp_path):
    doc = tmp_path / "empty.txt"
    doc.write_text("x")
    with patch("local_rag.loader.load_document", return_value="text"), \
         patch("local_rag.loader.chunk_text", return_value=[]):
        result = runner.invoke(main, ["add", str(doc)])
    assert result.exit_code == 0


def test_add_cmd_embed_error_exits_1(runner, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("x")
    with patch("local_rag.loader.load_document", return_value="text"), \
         patch("local_rag.loader.chunk_text", return_value=["chunk"]), \
         patch("local_rag.embedder.embed_texts", side_effect=EmbedError("no connection")):
        result = runner.invoke(main, ["add", str(doc)])
    assert result.exit_code == 1


def test_add_cmd_success(runner, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("x")
    with patch("local_rag.loader.load_document", return_value="text"), \
         patch("local_rag.loader.chunk_text", return_value=["chunk1", "chunk2"]), \
         patch("local_rag.embedder.embed_texts", return_value=[[0.1], [0.2]]), \
         patch("local_rag.store.add_chunks", return_value=2):
        result = runner.invoke(main, ["add", str(doc)])
    assert result.exit_code == 0


def test_add_cmd_reports_duplicates(runner, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("x")
    with patch("local_rag.loader.load_document", return_value="text"), \
         patch("local_rag.loader.chunk_text", return_value=["c1", "c2"]), \
         patch("local_rag.embedder.embed_texts", return_value=[[0.1], [0.2]]), \
         patch("local_rag.store.add_chunks", return_value=1):
        result = runner.invoke(main, ["add", str(doc)])
    assert result.exit_code == 0


def test_add_cmd_multiple_files(runner, tmp_path):
    doc1 = tmp_path / "a.txt"
    doc2 = tmp_path / "b.txt"
    doc1.write_text("x")
    doc2.write_text("y")
    with patch("local_rag.loader.load_document", return_value="text"), \
         patch("local_rag.loader.chunk_text", return_value=["chunk"]), \
         patch("local_rag.embedder.embed_texts", return_value=[[0.1]]), \
         patch("local_rag.store.add_chunks", return_value=1) as mock_add:
        result = runner.invoke(main, ["add", str(doc1), str(doc2)])
    assert result.exit_code == 0
    assert mock_add.call_count == 2


# ---------------------------------------------------------------------------
# ask command
# ---------------------------------------------------------------------------


def test_ask_cmd_no_sources_exits_1(runner):
    with patch("local_rag.store.list_sources", return_value=[]):
        result = runner.invoke(main, ["ask", "What?"])
    assert result.exit_code == 1


def test_ask_cmd_embed_error_exits_1(runner):
    with patch("local_rag.store.list_sources", return_value=["doc.txt"]), \
         patch("local_rag.embedder.embed_texts", side_effect=EmbedError("no connection")):
        result = runner.invoke(main, ["ask", "What?"])
    assert result.exit_code == 1


def test_ask_cmd_no_chunks_exits_1(runner):
    with patch("local_rag.store.list_sources", return_value=["doc.txt"]), \
         patch("local_rag.embedder.embed_texts", return_value=[[0.1]]), \
         patch("local_rag.store.query", return_value=[]):
        result = runner.invoke(main, ["ask", "What?"])
    assert result.exit_code == 1


def test_ask_cmd_llm_connection_error_exits_1(runner):
    chunks = [{"text": "ctx", "source": "doc.txt", "distance": 0.1}]
    with patch("local_rag.store.list_sources", return_value=["doc.txt"]), \
         patch("local_rag.embedder.embed_texts", return_value=[[0.1]]), \
         patch("local_rag.store.query", return_value=chunks), \
         patch("local_rag.llm.answer", side_effect=ConnectionError("no llm")):
        result = runner.invoke(main, ["ask", "What?"])
    assert result.exit_code == 1


def test_ask_cmd_llm_runtime_error_exits_1(runner):
    chunks = [{"text": "ctx", "source": "doc.txt", "distance": 0.1}]
    with patch("local_rag.store.list_sources", return_value=["doc.txt"]), \
         patch("local_rag.embedder.embed_texts", return_value=[[0.1]]), \
         patch("local_rag.store.query", return_value=chunks), \
         patch("local_rag.llm.answer", side_effect=RuntimeError("bad response")):
        result = runner.invoke(main, ["ask", "What?"])
    assert result.exit_code == 1


def test_ask_cmd_success(runner):
    chunks = [{"text": "ctx", "source": "doc.txt", "distance": 0.1}]
    with patch("local_rag.store.list_sources", return_value=["doc.txt"]), \
         patch("local_rag.embedder.embed_texts", return_value=[[0.1]]), \
         patch("local_rag.store.query", return_value=chunks), \
         patch("local_rag.llm.answer", return_value="The answer."):
        result = runner.invoke(main, ["ask", "What?"])
    assert result.exit_code == 0


def test_ask_cmd_show_sources_flag_succeeds(runner):
    chunks = [{"text": "ctx", "source": "/path/doc.txt", "distance": 0.12}]
    with patch("local_rag.store.list_sources", return_value=["doc.txt"]), \
         patch("local_rag.embedder.embed_texts", return_value=[[0.1]]), \
         patch("local_rag.store.query", return_value=chunks), \
         patch("local_rag.llm.answer", return_value="The answer."):
        result = runner.invoke(main, ["ask", "What?", "--show-sources"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# remove command
# ---------------------------------------------------------------------------


def test_remove_cmd_not_found_exits_1(runner):
    with patch("local_rag.store.list_sources", return_value=[]):
        result = runner.invoke(main, ["remove", "doc.txt"])
    assert result.exit_code == 1


def test_remove_cmd_ambiguous_match_exits_1(runner):
    with patch("local_rag.store.list_sources", return_value=["/path/doc.txt", "/path/doc2.txt"]):
        result = runner.invoke(main, ["remove", "doc"])
    assert result.exit_code == 1


def test_remove_cmd_yes_flag_deletes_source(runner):
    with patch("local_rag.store.list_sources", return_value=["/path/to/doc.txt"]), \
         patch("local_rag.store.delete_source", return_value=3) as mock_del:
        result = runner.invoke(main, ["remove", "doc.txt", "--yes"])
    assert result.exit_code == 0
    mock_del.assert_called_once_with("/path/to/doc.txt")


def test_remove_cmd_short_yes_flag(runner):
    with patch("local_rag.store.list_sources", return_value=["/path/to/doc.txt"]), \
         patch("local_rag.store.delete_source", return_value=1):
        result = runner.invoke(main, ["remove", "doc.txt", "-y"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# clear command
# ---------------------------------------------------------------------------


def test_clear_cmd_yes_flag_store_exists(runner, tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    monkeypatch.setattr("local_rag.cli.DB_DIR", db_dir)

    result = runner.invoke(main, ["clear", "--yes"])

    assert result.exit_code == 0
    assert not db_dir.exists()


def test_clear_cmd_yes_flag_store_already_empty(runner, tmp_path, monkeypatch):
    db_dir = tmp_path / "nonexistent_db"
    monkeypatch.setattr("local_rag.cli.DB_DIR", db_dir)

    result = runner.invoke(main, ["clear", "--yes"])

    assert result.exit_code == 0
