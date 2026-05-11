from unittest.mock import MagicMock
from core.pageindex.enrichment.base import BaseEnricher


def _make_llm(token_count=None):
    """Fake LLM client whose count_tokens returns token_count."""
    llm = MagicMock()
    llm.count_tokens = MagicMock(side_effect=lambda t: token_count if token_count is not None else len(t))
    return llm


class _FakeEncoding:
    """Minimal tiktoken-like encoding stub."""
    def encode(self, text):
        return list(text.encode("utf-8"))  # 1 byte = 1 "token" for test purposes

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


class TestTruncateToTokens:
    def test_returns_unchanged_when_under_budget(self):
        llm = _make_llm(token_count=50)
        enricher = BaseEnricher(llm)
        text = "hello world"
        assert enricher.truncate_to_tokens(text, 100) == text

    def test_returns_unchanged_when_exactly_at_budget(self):
        llm = _make_llm(token_count=100)
        enricher = BaseEnricher(llm)
        text = "exactly at budget"
        assert enricher.truncate_to_tokens(text, 100) == text

    def test_truncates_via_encoding_when_over_budget(self):
        llm = MagicMock()
        # count_tokens says 200, but we only want 5 "tokens" (bytes here)
        llm.count_tokens = MagicMock(return_value=200)
        llm.encoding = _FakeEncoding()
        enricher = BaseEnricher(llm)
        text = "hello world truncated text here"
        result = enricher.truncate_to_tokens(text, 5)
        # Should be the first 5 bytes decoded
        assert result == text.encode("utf-8")[:5].decode("utf-8", errors="replace")
        assert len(result) <= 5

    def test_ollama_heuristic_fallback(self):
        """When llm.encoding is absent, fall back to text[:max_tokens * 4]."""
        llm = MagicMock()
        llm.count_tokens = MagicMock(return_value=999)
        del llm.encoding  # ensure getattr returns None
        enricher = BaseEnricher(llm)
        text = "a" * 100
        result = enricher.truncate_to_tokens(text, 10)
        assert result == "a" * 40  # 10 * 4
