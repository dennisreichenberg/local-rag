"""Ollama embeddings via /api/embed."""

import httpx

from .config import EMBED_MODEL_DEFAULT, OLLAMA_DEFAULT_URL


class EmbedError(Exception):
    pass


def embed_texts(
    texts: list[str],
    *,
    model: str = EMBED_MODEL_DEFAULT,
    base_url: str = OLLAMA_DEFAULT_URL,
    timeout: float = 120.0,
) -> list[list[float]]:
    if not texts:
        return []
    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            response = client.post(
                "/api/embed",
                json={"model": model, "input": texts},
            )
            response.raise_for_status()
            return response.json()["embeddings"]
    except httpx.ConnectError as e:
        raise EmbedError(f"Cannot connect to Ollama at {base_url}") from e
    except httpx.HTTPStatusError as e:
        raise EmbedError(f"Ollama embed error {e.response.status_code}: {e.response.text[:200]}") from e
    except KeyError as e:
        raise EmbedError("Unexpected response from Ollama embed API") from e
