# local-rag

[![PyPI version](https://img.shields.io/pypi/v/ollama-local-rag)](https://pypi.org/project/ollama-local-rag/)
[![CI](https://github.com/dennisreichenberg/local-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/dennisreichenberg/local-rag/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/pypi/pyversions/ollama-local-rag)](https://pypi.org/project/ollama-local-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Ask questions about your documents using local LLMs — no cloud, no API keys, your data stays on your machine.

```
$ rag add research-paper.pdf annual-report.docx notes.md
Loading research-paper.pdf…
  128 chunks → embedding with nomic-embed-text…
  Added 128 new chunks
Loading annual-report.docx…
  94 chunks → embedding with nomic-embed-text…
  Added 94 new chunks

$ rag ask "What were the main revenue drivers in Q3?"

╭─ Answer ────────────────────────────────────────────────────────╮
│ Based on the annual report, the main revenue drivers in Q3      │
│ were cloud services (+34% YoY) and professional services…      │
│ [Source: annual-report.docx]                                    │
╰─────────────────────────────────────────────────────────────────╯
```

## Features

- **Local-first** — Ollama for embeddings + chat, ChromaDB for vector storage
- **Multiple formats** — PDF, DOCX, Markdown, plain text, RST
- **Persistent store** — add documents once, query forever
- **Source filtering** — restrict questions to specific files
- **Smart chunking** — overlapping word-based chunks for better context
- **Rich terminal UI** — markdown-rendered answers, source tables

## Requirements

- Python ≥ 3.10
- [Ollama](https://ollama.com) running locally
- Embedding model: `ollama pull nomic-embed-text`
- Chat model: `ollama pull mistral`

## Installation

```bash
pip install local-rag
```

Or from source:

```bash
git clone https://github.com/dennisreichenberg/local-rag
cd local-rag
pip install -e ".[dev]"
```

## Quick Start

```bash
# 1. Start Ollama (if not already running)
ollama serve

# 2. Pull required models
ollama pull nomic-embed-text
ollama pull mistral

# 3. Add documents
rag add report.pdf notes.md

# 4. Ask questions
rag ask "Summarize the key points"
rag ask "What are the risks mentioned?" --show-sources
```

## Commands

### `rag add <files...>`

Ingest one or more documents into the vector store. Supports `.pdf`, `.docx`, `.txt`, `.md`, `.rst`.

```bash
rag add report.pdf notes.md docs/
rag add report.pdf --embed-model nomic-embed-text --chunk-size 256
```

### `rag ask <question>`

Ask a question. Retrieves the most relevant chunks and sends them to the LLM.

```bash
rag ask "What is the conclusion?"
rag ask "Explain the architecture" --chat-model llama3 --top-k 8
rag ask "What risks are mentioned?" --source report.pdf --show-sources
```

### `rag list`

Show all ingested documents.

### `rag remove <source>`

Remove a document (and all its chunks) from the store. Supports partial path matching.

```bash
rag remove report.pdf
```

### `rag clear`

Remove everything from the store.

## Options

| Command | Option | Default | Description |
|---------|--------|---------|-------------|
| `add` | `--embed-model` | `nomic-embed-text` | Ollama embedding model |
| `add` | `--chunk-size` | `512` | Words per chunk |
| `add` | `--chunk-overlap` | `64` | Overlap between chunks |
| `ask` | `--chat-model` | `mistral` | Ollama chat model |
| `ask` | `--embed-model` | `nomic-embed-text` | Ollama embedding model |
| `ask` | `--top-k` | `5` | Chunks to retrieve |
| `ask` | `--source` | | Filter by source file |
| `ask` | `--show-sources` | | Show retrieved chunks |
| All | `--host` | `http://localhost:11434` | Ollama base URL |

## How it works

```
Document → Chunking → Ollama Embeddings → ChromaDB
                                              ↓
Question → Ollama Embeddings → Vector Search → Top-K Chunks → Ollama LLM → Answer
```

1. **Ingestion**: documents are split into overlapping chunks, embedded via Ollama (`nomic-embed-text`), and stored in a local ChromaDB database (`~/.local/share/local-rag/`)
2. **Retrieval**: your question is embedded, then the closest chunks are retrieved via cosine similarity
3. **Generation**: the retrieved chunks + question are sent to an Ollama chat model, which answers strictly from the provided context

## Data storage

All data is stored locally at `~/.local/share/local-rag/chroma/`. No data leaves your machine.

To move or backup your store, copy that directory.

## License

MIT — see [LICENSE](LICENSE)
