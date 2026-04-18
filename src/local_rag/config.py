"""Configuration and paths for local-rag."""

from pathlib import Path

OLLAMA_DEFAULT_URL = "http://localhost:11434"
EMBED_MODEL_DEFAULT = "nomic-embed-text"
CHAT_MODEL_DEFAULT = "mistral"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5
DB_DIR = Path.home() / ".local" / "share" / "local-rag" / "chroma"
