"""Tests for the store module using a temp-dir-isolated ChromaDB."""

import pytest

from local_rag.store import add_chunks, delete_source, list_sources, query


@pytest.fixture()
def db(tmp_path):
    """Provide a fresh, isolated DB directory for each test."""
    return tmp_path / "test_chroma"


def test_add_chunks_returns_added_count(db):
    count = add_chunks(
        ["hello world", "foo bar"], [[0.1, 0.2], [0.3, 0.4]], source="doc.txt", db_dir=db
    )
    assert count == 2


def test_add_chunks_deduplication_skips_existing(db):
    add_chunks(["chunk"], [[0.1, 0.2]], source="doc.txt", db_dir=db)
    count = add_chunks(["chunk"], [[0.1, 0.2]], source="doc.txt", db_dir=db)
    assert count == 0


def test_add_chunks_different_sources_not_deduplicated(db):
    add_chunks(["chunk"], [[0.1, 0.2]], source="a.txt", db_dir=db)
    count = add_chunks(["chunk"], [[0.1, 0.2]], source="b.txt", db_dir=db)
    assert count == 1


def test_list_sources_empty_collection(db):
    assert list_sources(db_dir=db) == []


def test_list_sources_returns_sorted_unique(db):
    add_chunks(["c1", "c2"], [[0.1, 0.2], [0.3, 0.4]], source="b.txt", db_dir=db)
    add_chunks(["c3"], [[0.5, 0.6]], source="a.txt", db_dir=db)
    sources = list_sources(db_dir=db)
    assert sources == ["a.txt", "b.txt"]


def test_list_sources_no_duplicates_per_source(db):
    add_chunks(
        ["c1", "c2", "c3"], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], source="doc.txt", db_dir=db
    )
    sources = list_sources(db_dir=db)
    assert sources.count("doc.txt") == 1


def test_query_returns_matching_results(db):
    add_chunks(["hello world"], [[0.9, 0.1]], source="doc.txt", db_dir=db)
    results = query([0.9, 0.1], top_k=5, db_dir=db)
    assert len(results) == 1
    assert results[0]["text"] == "hello world"
    assert results[0]["source"] == "doc.txt"
    assert "distance" in results[0]


def test_query_with_source_filter(db):
    add_chunks(["from a1", "from a2"], [[1.0, 0.0], [0.9, 0.1]], source="a.txt", db_dir=db)
    add_chunks(["from b"], [[0.0, 1.0]], source="b.txt", db_dir=db)
    results = query([1.0, 0.0], top_k=10, source_filter="a.txt", db_dir=db)
    assert all(r["source"] == "a.txt" for r in results)


def test_delete_source_returns_deleted_count(db):
    add_chunks(["c1", "c2"], [[0.1, 0.2], [0.3, 0.4]], source="doc.txt", db_dir=db)
    deleted = delete_source("doc.txt", db_dir=db)
    assert deleted == 2


def test_delete_source_nonexistent_returns_zero(db):
    assert delete_source("nonexistent.txt", db_dir=db) == 0


def test_delete_source_removes_from_list(db):
    add_chunks(["chunk"], [[0.1, 0.2]], source="doc.txt", db_dir=db)
    delete_source("doc.txt", db_dir=db)
    assert list_sources(db_dir=db) == []


def test_delete_source_only_removes_matching(db):
    add_chunks(["c1"], [[0.1, 0.2]], source="a.txt", db_dir=db)
    add_chunks(["c2"], [[0.3, 0.4]], source="b.txt", db_dir=db)
    delete_source("a.txt", db_dir=db)
    assert list_sources(db_dir=db) == ["b.txt"]
