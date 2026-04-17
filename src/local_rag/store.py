"""ChromaDB vector store wrapper."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from .config import DB_DIR, TOP_K


def _client(db_dir: Path = DB_DIR) -> chromadb.ClientAPI:
    db_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(db_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection(name: str = "docs", db_dir: Path = DB_DIR) -> chromadb.Collection:
    return _client(db_dir).get_or_create_collection(name)


def add_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source: str,
    collection_name: str = "docs",
    db_dir: Path = DB_DIR,
) -> int:
    col = _collection(collection_name, db_dir)
    existing_ids = set(col.get(where={"source": source})["ids"])

    ids, docs, embeds, metas = [], [], [], []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{source}::{i}"
        if chunk_id not in existing_ids:
            ids.append(chunk_id)
            docs.append(chunk)
            embeds.append(emb)
            metas.append({"source": source, "chunk_index": i})

    if ids:
        col.add(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)
    return len(ids)


def query(
    embedding: list[float],
    top_k: int = TOP_K,
    source_filter: str | None = None,
    collection_name: str = "docs",
    db_dir: Path = DB_DIR,
) -> list[dict]:
    col = _collection(collection_name, db_dir)
    where = {"source": source_filter} if source_filter else None
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(top_k, col.count() or 1),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({"text": doc, "source": meta.get("source", ""), "distance": dist})
    return out


def list_sources(collection_name: str = "docs", db_dir: Path = DB_DIR) -> list[str]:
    col = _collection(collection_name, db_dir)
    metas = col.get(include=["metadatas"])["metadatas"]
    return sorted({m.get("source", "") for m in metas if m.get("source")})


def delete_source(
    source: str, collection_name: str = "docs", db_dir: Path = DB_DIR
) -> int:
    col = _collection(collection_name, db_dir)
    ids = col.get(where={"source": source})["ids"]
    if ids:
        col.delete(ids=ids)
    return len(ids)
