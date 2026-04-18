"""Ollama chat completion for answering questions from retrieved context."""

import httpx

from .config import CHAT_MODEL_DEFAULT, OLLAMA_DEFAULT_URL

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based strictly on the provided context.
If the answer is not contained in the context, say so clearly — do not make up information.
Cite the source file when relevant.\
"""


def answer(
    question: str,
    context_chunks: list[dict],
    *,
    model: str = CHAT_MODEL_DEFAULT,
    base_url: str = OLLAMA_DEFAULT_URL,
    timeout: float = 120.0,
) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    user_message = f"Context:\n{context_text}\n\nQuestion: {question}"

    try:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            response = client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
    except httpx.ConnectError as e:
        raise ConnectionError(f"Cannot connect to Ollama at {base_url}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama error {e.response.status_code}: {e.response.text[:200]}") from e
