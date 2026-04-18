"""Tests for the embedder module."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from local_rag.embedder import EmbedError, embed_texts


def test_embed_empty_list_returns_empty():
    assert embed_texts([]) == []


def test_embed_success_returns_embeddings():
    with patch("local_rag.embedder.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        result = embed_texts(["hello world"], model="test-model")

    assert result == [[0.1, 0.2, 0.3]]


def test_embed_sends_correct_payload():
    with patch("local_rag.embedder.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {"embeddings": [[0.5], [0.6]]}

        embed_texts(["text1", "text2"], model="mymodel")

    _, kwargs = instance.post.call_args
    assert kwargs["json"]["model"] == "mymodel"
    assert kwargs["json"]["input"] == ["text1", "text2"]


def test_embed_posts_to_correct_endpoint():
    with patch("local_rag.embedder.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {"embeddings": [[0.1]]}

        embed_texts(["hello"])

    args, _ = instance.post.call_args
    assert args[0] == "/api/embed"


def test_embed_connect_error_raises_embed_error():
    with patch("local_rag.embedder.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.side_effect = httpx.ConnectError("refused")

        with pytest.raises(EmbedError, match="Cannot connect"):
            embed_texts(["hello"])


def test_embed_http_status_error_includes_status_code():
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"

    with patch("local_rag.embedder.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=mock_request, response=mock_response
        )

        with pytest.raises(EmbedError, match="404"):
            embed_texts(["hello"])


def test_embed_missing_embeddings_key_raises_embed_error():
    with patch("local_rag.embedder.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {"wrong_key": []}

        with pytest.raises(EmbedError, match="Unexpected response"):
            embed_texts(["hello"])


def test_embed_multiple_texts_returns_multiple_embeddings():
    with patch("local_rag.embedder.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        }

        result = embed_texts(["a", "b", "c"])

    assert len(result) == 3
    assert result[1] == [0.3, 0.4]
