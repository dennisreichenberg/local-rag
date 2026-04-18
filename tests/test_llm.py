"""Tests for the llm module."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from local_rag.llm import answer

SAMPLE_CHUNKS = [
    {"source": "doc.txt", "text": "The sky is blue."},
    {"source": "doc2.txt", "text": "Water is wet."},
]


def test_answer_returns_stripped_content():
    with patch("local_rag.llm.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {
            "message": {"content": "  The answer.  "}
        }
        result = answer("What?", SAMPLE_CHUNKS)

    assert result == "The answer."


def test_answer_sends_question_and_context():
    with patch("local_rag.llm.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {"message": {"content": "ok"}}

        answer("My question", SAMPLE_CHUNKS, model="llama3")

    _, kwargs = instance.post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "llama3"
    assert payload["stream"] is False
    user_msg = payload["messages"][1]["content"]
    assert "My question" in user_msg
    assert "doc.txt" in user_msg
    assert "The sky is blue." in user_msg


def test_answer_posts_to_chat_endpoint():
    with patch("local_rag.llm.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {"message": {"content": "x"}}

        answer("question", SAMPLE_CHUNKS)

    args, _ = instance.post.call_args
    assert args[0] == "/api/chat"


def test_answer_system_prompt_is_included():
    with patch("local_rag.llm.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {"message": {"content": "x"}}

        answer("question", SAMPLE_CHUNKS)

    _, kwargs = instance.post.call_args
    messages = kwargs["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert len(messages[0]["content"]) > 0


def test_answer_context_separates_chunks():
    with patch("local_rag.llm.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {"message": {"content": "x"}}

        answer("q", SAMPLE_CHUNKS)

    _, kwargs = instance.post.call_args
    user_msg = kwargs["json"]["messages"][1]["content"]
    assert "doc2.txt" in user_msg
    assert "Water is wet." in user_msg


def test_answer_empty_context_chunks():
    with patch("local_rag.llm.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.json.return_value = {"message": {"content": "No context."}}

        result = answer("question", [])

    assert result == "No context."


def test_answer_connect_error_raises_connection_error():
    with patch("local_rag.llm.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ConnectionError, match="Cannot connect"):
            answer("question", SAMPLE_CHUNKS)


def test_answer_http_status_error_raises_runtime_error():
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Error"

    with patch("local_rag.llm.httpx.Client") as MockClient:
        instance = MockClient.return_value.__enter__.return_value
        instance.post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=mock_request, response=mock_response
        )

        with pytest.raises(RuntimeError, match="500"):
            answer("question", SAMPLE_CHUNKS)
