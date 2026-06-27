"""Tests for translation mode routing (element vs block)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.translators.block_translator import BlockTranslator
from services.translators.element_translator import ElementTranslator


class TestTranslationRouting:
    def test_threshold_logic_selects_block(self):
        mock_settings = MagicMock()
        mock_settings.translation_block_merge = True
        mock_settings.translation_element_max = 500

        count = 600
        use_blocks = (
            mock_settings.translation_block_merge
            and count > mock_settings.translation_element_max
        )
        assert use_blocks is True

    def test_threshold_logic_selects_element(self):
        mock_settings = MagicMock()
        mock_settings.translation_block_merge = True
        mock_settings.translation_element_max = 500

        count = 10
        use_blocks = (
            mock_settings.translation_block_merge
            and count > mock_settings.translation_element_max
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
