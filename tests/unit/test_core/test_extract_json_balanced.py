"""extract_json must parse balanced JSON, not greedy-regex from first `{` to
last `}` — LLM responses with two JSON objects or trailing prose made the
greedy span unparseable and the whole extraction fail.
"""

import pytest

from core.pageindex.llm.llm_client_base import BaseLLMClient


class _Client(BaseLLMClient):
    async def chat_completion(self, prompt, chat_history=None, **kwargs):  # pragma: no cover
        return ""

    async def chat_completion_with_finish_reason(
        self, prompt, chat_history=None, **kwargs
    ):  # pragma: no cover
        return "", "stop"

    def count_tokens(self, text):  # pragma: no cover
        return len(text) // 4


@pytest.fixture
def client():
    return _Client(model="test")


def test_two_objects_with_prose_parses_first(client):
    content = 'Here you go: {"keywords": ["a", "b"]} and also maybe {"other": 1} — hope it helps!'
    assert client.extract_json(content) == {"keywords": ["a", "b"]}


def test_object_with_trailing_brace_in_prose(client):
    content = '{"ok": true} (note: use {curly} syntax elsewhere)'
    assert client.extract_json(content) == {"ok": True}


def test_array_with_trailing_prose(client):
    content = 'Result: [{"id": 1}, {"id": 2}] — done.'
    assert client.extract_json(content) == [{"id": 1}, {"id": 2}]


def test_fenced_json_still_works(client):
    content = 'Sure!\n```json\n{"x": 1}\n```\nThanks.'
    assert client.extract_json(content) == {"x": 1}


def test_plain_json_still_works(client):
    assert client.extract_json('{"y": 2}') == {"y": 2}


def test_no_json_raises(client):
    import json

    with pytest.raises(json.JSONDecodeError):
        client.extract_json("no structured data here at all")
