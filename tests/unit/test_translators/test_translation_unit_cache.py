"""Persistent per-run translation cache (MinIO-backed): after a crash or
retry, every already-translated unit must be a cache hit so the run resumes
instead of re-translating a multi-hour book from zero. Shared by every
translation mode because they all funnel through StructuredTranslator.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.translators._cache import TranslationUnitCache


class FakeStorage:
    def __init__(self):
        self.objects: dict = {}

    def put_bytes(self, key, data, **kwargs):
        self.objects[key] = data

    def get_bytes(self, key):
        return self.objects[key]

    def list_keys(self, prefix):
        return [k for k in self.objects if k.startswith(prefix)]

    def delete_prefix(self, prefix):
        for k in list(self.objects):
            if k.startswith(prefix):
                del self.objects[k]


class TestTranslationUnitCache:
    def test_roundtrip_and_scope(self):
        storage = FakeStorage()
        cache = TranslationUnitCache("DOC_1", "TRN_1", target_lang="vi", storage=storage)
        cache.load()
        assert cache.get("text", "Hello") is None
        cache.put("text", "Hello", "Xin chào")
        assert cache.get("text", "Hello") == "Xin chào"
        # persisted under the run prefix
        assert all(k.startswith("documents/DOC_1/translations/TRN_1/") for k in storage.objects)

    def test_survives_restart_via_load(self):
        storage = FakeStorage()
        c1 = TranslationUnitCache("DOC_1", "TRN_1", target_lang="vi", storage=storage)
        c1.load()
        c1.put("text", "Hello", "Xin chào")
        c1.put("title", "Chapter 1", "Chương 1")

        c2 = TranslationUnitCache("DOC_1", "TRN_1", target_lang="vi", storage=storage)
        c2.load()
        assert c2.get("text", "Hello") == "Xin chào"
        assert c2.get("title", "Chapter 1") == "Chương 1"

    def test_kind_and_language_partition_keys(self):
        storage = FakeStorage()
        cache = TranslationUnitCache("DOC_1", "TRN_1", target_lang="vi", storage=storage)
        cache.load()
        cache.put("text", "Hello", "Xin chào")
        assert cache.get("title", "Hello") is None

        cache_fr = TranslationUnitCache("DOC_1", "TRN_1", target_lang="fr", storage=storage)
        cache_fr.load()
        assert cache_fr.get("text", "Hello") is None


@pytest.mark.asyncio
async def test_translator_resumes_from_cache():
    """Second run over the same units must make ZERO LLM calls."""
    from core.pageindex.enrichment.translator import StructuredTranslator

    storage = FakeStorage()

    def make_translator():
        llm = AsyncMock()
        llm.chat_completion_with_finish_reason = AsyncMock(return_value=("bản dịch", "stop"))
        llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
        llm.encoding = None
        cache = TranslationUnitCache("DOC_1", "TRN_1", target_lang="vi", storage=storage)
        cache.load()
        t = StructuredTranslator(llm_client=llm, source_lang="en", target_lang="vi")
        t.unit_cache = cache
        return t, llm

    t1, llm1 = make_translator()
    await t1.translate_text("A sentence to translate.")
    await t1.translate_title("A Title")
    assert llm1.chat_completion_with_finish_reason.await_count == 2

    # fresh translator + fresh memo, same storage → all cache hits
    t2, llm2 = make_translator()
    out = await t2.translate_text("A sentence to translate.")
    title = await t2.translate_title("A Title")
    assert out == "bản dịch" and title == "bản dịch"
    assert llm2.chat_completion_with_finish_reason.await_count == 0


@pytest.mark.asyncio
async def test_degraded_output_not_cached():
    from core.pageindex.enrichment.translator import StructuredTranslator

    storage = FakeStorage()
    llm = AsyncMock()
    llm.chat_completion_with_finish_reason = AsyncMock(return_value=("", "stop"))
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    cache = TranslationUnitCache("DOC_1", "TRN_1", target_lang="vi", storage=storage)
    cache.load()
    t = StructuredTranslator(llm_client=llm, source_lang="en", target_lang="vi")
    t.unit_cache = cache

    out = await t.translate_text("Some source text.")
    assert out == "Some source text."  # degraded to source
    assert cache.get("text", "Some source text.") is None  # bad result not frozen
