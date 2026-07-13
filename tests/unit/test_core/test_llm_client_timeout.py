"""Every LLM HTTP client must carry an explicit request timeout — without one,
a stalled backend (llama.cpp queue wedged, network partition) parks a worker
slot forever. The overlay path compounds this with paragraph-level retries.
"""

from unittest.mock import MagicMock, patch

from config.settings import settings


def test_openai_async_client_sets_timeout():
    with patch("core.pageindex.llm.openai_client.AsyncOpenAI") as mock_cls:
        from core.pageindex.llm.openai_client import OpenAIClient

        OpenAIClient(model="qwen3.5-9b", api_key="k", base_url="http://x:5011/v1")

    kwargs = mock_cls.call_args.kwargs
    assert kwargs.get("timeout") == settings.ai_request_timeout_seconds


def test_overlay_sync_client_sets_timeout():
    with patch("openai.OpenAI") as mock_cls:
        from core.pdf_overlay.llm_adapter import OverlayLLMAdapter

        OverlayLLMAdapter(source_lang="en", target_lang="vi")

    kwargs = mock_cls.call_args.kwargs
    assert kwargs.get("timeout") == settings.ai_request_timeout_seconds
