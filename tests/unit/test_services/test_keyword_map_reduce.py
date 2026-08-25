"""The LLM must read the whole document, not one 8k window of it.

The final re-rank grounds against `build_pipeline_doc_sample`, capped at
`min(ai_chunk_tokens - 2000, 8000)` tokens. On an 816-page book that is roughly
2% of the text, so every term from the other 98% has to arrive through the
candidate list or it is unreachable — no re-ranker can pick what it never sees.

A statistical tier alone cannot close that: it ranks surface features. This map
pass chunks the full text and asks the model for terms per chunk, reusing the
llama-server already running. No new model is loaded — this session went the
other way, splitting extraction out precisely to keep models off the GPU.

The cost ceiling is real and must be visible: a very long document is truncated
to `_MAX_MAP_CHUNKS`, and that has to be logged. A silent cap reads as "the
whole document was covered" when it was not.
"""

import json
import logging
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.keyword_service import _MAX_MAP_CHUNKS, KeywordService


def _parse(raw, **kwargs):
    """Stand-in for the real client's `extract_json`: parse, or raise."""
    return json.loads(raw)


def _llm(payloads):
    """An LLM whose successive chunk calls return successive payloads."""
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(side_effect=list(payloads))
    llm.extract_json = MagicMock(side_effect=_parse)
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    return llm


def _svc():
    svc = KeywordService.__new__(KeywordService)
    svc._progress = MagicMock()
    return svc


PARA = "Đoạn nội dung kỹ thuật về bộ nhớ đệm và đường ống lệnh. " * 40
# Comfortably larger than one PARA so each paragraph lands in its own chunk
# and the marker survives — chunk_text falls back to a sentence split otherwise.
_CHUNK_TOKENS = len(PARA) // 3


class TestMapPass:
    @pytest.mark.asyncio
    async def test_terms_from_every_chunk_are_collected(self):
        text = "\n\n".join([PARA] * 3)
        llm = _llm(
            [
                '["bộ nhớ đệm", "đường ống lệnh"]',
                '["dự đoán rẽ nhánh"]',
                '["kiến trúc siêu vô hướng"]',
            ]
        )

        out = await _svc()._llm_candidates(llm, text, chunk_tokens=_CHUNK_TOKENS)

        names = [c["keyword"] for c in out]
        assert "bộ nhớ đệm" in names
        assert "dự đoán rẽ nhánh" in names
        assert "kiến trúc siêu vô hướng" in names

    @pytest.mark.asyncio
    async def test_duplicates_across_chunks_collapse(self):
        text = "\n\n".join([PARA] * 2)
        llm = _llm(['["bộ nhớ đệm"]', '["  Bộ Nhớ Đệm  "]'])

        out = await _svc()._llm_candidates(llm, text, chunk_tokens=_CHUNK_TOKENS)

        assert [c["keyword"] for c in out] == ["bộ nhớ đệm"]

    @pytest.mark.asyncio
    async def test_a_chunk_that_fails_does_not_sink_the_pass(self):
        """Keywords is non-critical; one bad chunk must not lose the other two."""
        text = "\n\n".join([PARA] * 3)
        llm = _llm(['["bộ nhớ đệm"]', RuntimeError("backend down"), '["đường ống lệnh"]'])

        out = await _svc()._llm_candidates(llm, text, chunk_tokens=_CHUNK_TOKENS)

        names = [c["keyword"] for c in out]
        assert "bộ nhớ đệm" in names and "đường ống lệnh" in names

    @pytest.mark.asyncio
    async def test_unparseable_output_is_skipped_not_fatal(self):
        text = "\n\n".join([PARA] * 2)
        llm = _llm(["I'm afraid I can't do that", '["bộ nhớ đệm"]'])

        out = await _svc()._llm_candidates(llm, text, chunk_tokens=_CHUNK_TOKENS)

        assert [c["keyword"] for c in out] == ["bộ nhớ đệm"]

    @pytest.mark.asyncio
    async def test_empty_text_makes_no_calls(self):
        llm = _llm([])

        assert await _svc()._llm_candidates(llm, "   ", chunk_tokens=100) == []
        llm.chat_completion.assert_not_awaited()


class TestCoverageIsSpreadNotTruncated:
    """Measured on N4.11.160: 61 chunks, cap 24 — `chunks[:24]` read the first
    39% of the book and nothing after it.

    The result was visible in §2.3: every term the map pass contributed was
    chapter-1 vocabulary (`Máy ảo`, `Ngôn ngữ máy`, `Phần mềm`, `Phần cứng`),
    while chapter-6/8 specifics the previous run had found — `bộ nhớ ảo`,
    `hệ thống RISC và CISC` — disappeared entirely. Reading a prefix made the
    keyword list MORE front-weighted, which is the opposite of why this pass
    exists.

    A cap is a sampling rate, not a stopping point: spread the chunks we can
    afford across the whole document.
    """

    @staticmethod
    def _sections_seen(llm):
        """Every SECTION-i marker that reached the model, in call order."""
        out = []
        for call in llm.chat_completion.await_args_list:
            out += [int(m) for m in re.findall(r"SECTION-(\d+)", call.args[0])]
        return out

    @pytest.mark.asyncio
    async def test_the_sample_spans_the_whole_document(self):
        n = _MAX_MAP_CHUNKS * 3
        text = "\n\n".join(f"SECTION-{i} " + PARA for i in range(n))
        llm = _llm(['["thuật ngữ"]'] * (n + 5))

        await _svc()._llm_candidates(llm, text, chunk_tokens=_CHUNK_TOKENS)

        seen = self._sections_seen(llm)
        assert llm.chat_completion.await_count == _MAX_MAP_CHUNKS
        # A prefix read stops around n/3; a spread sample reaches the end.
        assert max(seen) > n * 0.8, f"sample stops at {max(seen)} of {n}"
        assert min(seen) < n * 0.2, f"sample starts at {min(seen)}"

    @pytest.mark.asyncio
    async def test_a_document_under_the_cap_is_read_whole_and_in_order(self):
        n = 5
        text = "\n\n".join(f"SECTION-{i} " + PARA for i in range(n))
        llm = _llm(['["thuật ngữ"]'] * (n + 5))

        await _svc()._llm_candidates(llm, text, chunk_tokens=_CHUNK_TOKENS)

        assert self._sections_seen(llm) == list(range(n))


class TestCostCeiling:
    @pytest.mark.asyncio
    async def test_a_long_document_is_capped(self):
        text = "\n\n".join([PARA] * (_MAX_MAP_CHUNKS + 6))
        llm = _llm(['["thuật ngữ"]'] * (_MAX_MAP_CHUNKS + 6))

        await _svc()._llm_candidates(llm, text, chunk_tokens=_CHUNK_TOKENS)

        assert llm.chat_completion.await_count == _MAX_MAP_CHUNKS

    @pytest.mark.asyncio
    async def test_the_cap_is_logged_not_silent(self, caplog):
        """Dropping coverage silently reads as full coverage. Say it."""
        text = "\n\n".join([PARA] * (_MAX_MAP_CHUNKS + 6))
        llm = _llm(['["thuật ngữ"]'] * (_MAX_MAP_CHUNKS + 6))

        with caplog.at_level(logging.WARNING, logger="services.keyword_service"):
            await _svc()._llm_candidates(llm, text, chunk_tokens=_CHUNK_TOKENS)

        assert str(_MAX_MAP_CHUNKS) in caplog.text
