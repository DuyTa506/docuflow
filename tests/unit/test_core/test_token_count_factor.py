"""tiktoken undercounts the served model's tokenizer; budgets must not.

Live regression (E2E, DOC_014): keyword map chunks sized to 13 926 cl100k
tokens were 17 499 Gemma tokens — over the 16 384 per-slot context, so the
chunk was rejected and its keywords lost. Measured Gemma/cl100k ratio on the
E2E books: 0.52–1.26 (English prose with code highest).
"""

import tiktoken

from core.pageindex.enrichment.base import BaseEnricher
from core.pageindex.llm.openai_client import OpenAIClient

TEXT = "The switching regulator converts the input voltage efficiently. " * 200


def _client(factor):
    c = OpenAIClient.__new__(OpenAIClient)  # skip network-y __init__
    c.encoding = tiktoken.get_encoding("cl100k_base")
    c.token_count_factor = factor
    return c


def test_count_is_scaled_by_factor():
    raw = len(tiktoken.get_encoding("cl100k_base").encode(TEXT))
    assert _client(1.3).count_tokens(TEXT) >= raw * 1.3


def test_truncate_respects_scaled_budget():
    client = _client(1.3)
    out = BaseEnricher(client).truncate_to_tokens(TEXT, 100)
    assert client.count_tokens(out) <= 100


def test_factor_comes_from_settings(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "ai_token_count_factor", 1.4, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    client = OpenAIClient(model="gemma-4-26b", base_url="http://localhost:1/v1")
    assert client.token_count_factor == 1.4
