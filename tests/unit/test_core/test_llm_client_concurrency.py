"""Tests that BaseLLMClient bounds concurrent requests via a shared semaphore."""

import asyncio

import pytest

from core.pageindex.llm.openai_client import OpenAIClient


class _FakeCompletions:
    def __init__(self, delay: float, in_flight: list, peak: list):
        self._delay = delay
        self._in_flight = in_flight
        self._peak = peak

    async def create(self, **kwargs):
        self._in_flight.append(1)
        self._peak[0] = max(self._peak[0], len(self._in_flight))
        await asyncio.sleep(self._delay)
        self._in_flight.pop()

        class _Msg:
            content = "ok"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeAsyncOpenAI:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


class TestChatCompletionConcurrencyGate:
    @pytest.mark.asyncio
    async def test_peak_concurrency_never_exceeds_max_concurrent(self, monkeypatch):
        in_flight: list = []
        peak = [0]
        fake_completions = _FakeCompletions(delay=0.05, in_flight=in_flight, peak=peak)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = OpenAIClient(model="gpt-4o", max_concurrent=2)
        client.client = _FakeAsyncOpenAI(fake_completions)

        await asyncio.gather(*(client.chat_completion("hi") for _ in range(6)))

        assert peak[0] <= 2

    @pytest.mark.asyncio
    async def test_default_max_concurrent_is_four(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = OpenAIClient(model="gpt-4o")
        assert client._semaphore._value == 4
