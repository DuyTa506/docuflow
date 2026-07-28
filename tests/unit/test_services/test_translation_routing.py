"""Tests for translation mode routing (element vs block, overlay eligibility)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.translators.block_translator import BlockTranslator
from services.translators.element_translator import ElementTranslator
from utils.export_paths import translation_routing_allows_overlay


class TestOverlayEligibility:
    """`scanned == 0` was all-or-nothing.

    On N4.11.160 (816-page text-layer book) a handful of plate/blank pages fall
    under PDF_TEXT_THRESHOLD=50 chars and are labelled "scanned", which knocked
    the *entire* book off pdf_overlay into the flat tree path — the worst
    available mode. A scanned page simply has no text layer for the overlay to
    translate, so it costs nothing to leave it as-is.
    """

    def test_allows_overlay_for_fully_text_layer_pdf(self):
        assert translation_routing_allows_overlay(0, 816) is True

    def test_allows_overlay_with_a_few_scanned_pages(self):
        assert translation_routing_allows_overlay(3, 816) is True

    def test_rejects_overlay_above_ratio(self):
        assert translation_routing_allows_overlay(200, 816) is False

    def test_rejects_overlay_for_fully_scanned_document(self):
        assert translation_routing_allows_overlay(40, 40) is False

    def test_unknown_page_count_falls_back_to_strict_gate(self):
        assert translation_routing_allows_overlay(0, 0) is True
        assert translation_routing_allows_overlay(1, 0) is False


class TestTranslationRouting:
    def test_threshold_logic_selects_block(self):
        mock_settings = MagicMock()
        mock_settings.translation_block_merge = True
        mock_settings.translation_element_max = 500

        count = 600
        use_blocks = (
            mock_settings.translation_block_merge and count > mock_settings.translation_element_max
        )
        assert use_blocks is True

    def test_threshold_logic_selects_element(self):
        mock_settings = MagicMock()
        mock_settings.translation_block_merge = True
        mock_settings.translation_element_max = 500

        count = 10
        use_blocks = (
            mock_settings.translation_block_merge and count > mock_settings.translation_element_max
        )
        assert use_blocks is False

    @pytest.mark.asyncio
    async def test_block_translator_used_for_large_payloads(self):
        mock_translator = MagicMock()
        mock_translator.translate_text = AsyncMock(return_value="dich")
        mock_translator.translate_title = AsyncMock(return_value="tieu de")

        payloads = [
            {
                "page_number": 1,
                "sequence_order": i,
                "label": "text",
                "text_content": f"word {i}",
                "bbox": {"x1": 0, "y1": i * 25, "x2": 50, "y2": i * 25 + 20},
            }
            for i in range(600)
        ]

        result = await BlockTranslator(mock_translator).translate_payloads(payloads)

        assert result["translation_mode"] == "block_based"
        assert mock_translator.translate_text.await_count < len(payloads)

    @pytest.mark.asyncio
    async def test_element_translator_one_call_per_element(self):
        mock_translator = MagicMock()
        mock_translator.translate_text = AsyncMock(return_value="dich")
        mock_translator.translate_title = AsyncMock(return_value="tieu de")

        payloads = [
            {
                "page_number": 1,
                "sequence_order": i,
                "label": "text",
                "text_content": f"word {i}",
                "bbox": {"x1": 0, "y1": i * 25, "x2": 50, "y2": i * 25 + 20},
            }
            for i in range(3)
        ]

        with patch("services.translators.element_translator.settings") as mock_settings:
            mock_settings.translation_parallelism = 1
            result = await ElementTranslator(mock_translator).translate_payloads(payloads)

        assert result["translation_mode"] == "element_based"
        assert mock_translator.translate_text.await_count == 3
