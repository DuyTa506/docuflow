"""Translation output validation — empty, truncated, degenerate, or
wrong-language completions must be caught instead of saved as COMPLETED
(previously nothing inspected LLM output; retry fired only on exceptions).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.pageindex.enrichment.validation import validate_translation

VI = "vi"


class TestValidateTranslation:
    def test_valid_output_passes(self):
        r = validate_translation("Hello world", "Xin chào thế giới", VI)
        assert r.ok

    def test_empty_output_with_nonempty_source_fails(self):
        r = validate_translation("Hello world", "   ", VI)
        assert not r.ok and r.reason == "empty"

    def test_empty_source_always_passes(self):
        assert validate_translation("", "", VI).ok

    def test_finish_reason_length_fails_as_truncated(self):
        r = validate_translation("Hello", "Xin", VI, finish_reason="length")
        assert not r.ok and r.reason == "truncated"

    def test_degenerate_repetition_fails(self):
        out = "xin chào bạn nhé " * 40
        r = validate_translation("A long source paragraph " * 20, out, VI)
        assert not r.ok and r.reason == "degenerate"

    def test_length_ratio_explosion_fails(self):
        src = "Short source sentence but definitely above eighty characters long in total, yes."
        r = validate_translation(src, "word " * 400, VI)
        assert not r.ok and r.reason in ("length_ratio", "degenerate")

    def test_wrong_language_fails_for_long_output(self):
        src = "x" * 300
        out = " ".join(f"sentence number {i} with distinct wording throughout" for i in range(12))
        with patch(
            "core.pageindex.enrichment.validation.detect_source_language",
            return_value="en",
        ):
            r = validate_translation(src, out, VI)
        assert not r.ok and r.reason == "wrong_language"

    def test_short_output_skips_language_check(self):
        with patch(
            "core.pageindex.enrichment.validation.detect_source_language",
            return_value="en",
        ) as mock_detect:
            r = validate_translation("Tiêu đề", "Short title", VI)
        assert r.ok
        mock_detect.assert_not_called()


class TestTranslateTextValidated:
    def _translator(self, llm):
        from core.pageindex.enrichment.translator import StructuredTranslator

        return StructuredTranslator(
            llm_client=llm, source_lang="en", target_lang="vi", chunk_size=6000
        )

    @pytest.mark.asyncio
    async def test_garbage_output_retries_then_degrades_to_source(self):
        llm = AsyncMock()
        llm.chat_completion_with_finish_reason = AsyncMock(return_value=("", "stop"))
        llm.count_tokens = MagicMock(return_value=100)

        translator = self._translator(llm)
        source = "A meaningful source paragraph."
        out = await translator.translate_text(source)

        assert out == source  # degraded to source, not empty
        assert translator.degraded_units == 1
        assert llm.chat_completion_with_finish_reason.await_count == 3

    @pytest.mark.asyncio
    async def test_truncated_output_splits_and_retranslates(self):
        calls = []
        # Varied full-size output — proportional length, non-repeating.
        good_output = " ".join(f"câu dịch thứ {i} với nội dung riêng" for i in range(60))

        async def fake_completion(prompt, **kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                return ("một phần dịch bị cụt", "length")
            return (good_output, "stop")

        llm = AsyncMock()
        llm.chat_completion_with_finish_reason = AsyncMock(side_effect=fake_completion)
        llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
        llm.encoding = None

        translator = self._translator(llm)
        source = "First sentence here. " * 100
        with patch(
            "core.pageindex.enrichment.validation.detect_source_language",
            return_value="vi",
        ):
            out = await translator.translate_text(source)

        assert good_output in out
        assert translator.degraded_units == 0
        # first call truncated → at least two follow-up half calls
        assert llm.chat_completion_with_finish_reason.await_count >= 3
