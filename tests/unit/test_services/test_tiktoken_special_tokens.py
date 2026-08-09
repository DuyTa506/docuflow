"""Token counting must survive text that LOOKS like special tokens.

Regression (DOC_069, an LLM tech report): the document body literally
contained the string '<|endoftext|>' (AI papers discuss special tokens!).
tiktoken's encode() raises on special-token text by default, so
count_tokens / truncate_to_tokens crashed deterministically — the
HIERARCHICAL_SUMMARIZE stage failed all 5 retries (~28 min of RUNNING)
and, being a critical stage, took the whole digest to FAILED.
"""

import tiktoken

from core.pageindex.enrichment.base import BaseEnricher
from core.pageindex.llm.openai_client import OpenAIClient

SPECIAL = "Bài báo mô tả token đặc biệt <|endoftext|> và <|fim_prefix|> trong huấn luyện. " * 50


def _client():
    c = OpenAIClient.__new__(OpenAIClient)  # skip network-y __init__
    c.encoding = tiktoken.get_encoding("cl100k_base")
    return c


class TestSpecialTokenTextIsPlainText:
    def test_count_tokens_does_not_raise(self):
        assert _client().count_tokens(SPECIAL) > 0

    def test_truncate_to_tokens_does_not_raise_and_respects_budget(self):
        client = _client()
        enricher = BaseEnricher(client)
        out = enricher.truncate_to_tokens(SPECIAL, 100)
        assert client.count_tokens(out) <= 100
