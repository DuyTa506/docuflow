"""Tests for BlockTranslator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from services.translators.block_translator import BlockTranslator


def _payload(page, order, label, text, y1=20):
    return {
        "page_number": page,
        "sequence_order": order,
        "label": label,
        "text_content": text,
        "bbox": {"x1": 10, "y1": y1, "x2": 100, "y2": y1 + 20},
    }


class TestBlockTranslator:
    @pytest.mark.asyncio
    async def test_translates_merged_block_and_passthrough_image(self):
        translator = MagicMock()
        translator.translate_text = AsyncMock(return_value="Xin chao the gioi")
        translator.translate_title = AsyncMock(return_value="Tieu de")

        svc = BlockTranslator(translator)
        payloads = [
            _payload(1, 0, "text", "Hello", y1=10),
            _payload(1, 1, "text", "world", y1=30),
            _payload(1, 2, "image", "pic", y1=100),
        ]

        result = await svc.translate_payloads(payloads)

        assert result["translation_mode"] == "block_based"
        assert len(result["translated_elements"]) == 2
        assert result["translated_elements"][0]["text_content"] == "Xin chao the gioi"
        assert result["translated_elements"][0]["bbox"]["y1"] == 10
        assert result["translated_elements"][1]["label"] == "image"
        assert result["translated_elements"][1]["text_content"] == "pic"
        translator.translate_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_heading_uses_translate_title(self):
        translator = MagicMock()
        translator.translate_text = AsyncMock(return_value="ignored")
        translator.translate_title = AsyncMock(return_value="Tieu de")

        svc = BlockTranslator(translator)
        payloads = [_payload(1, 0, "title", "Hello", y1=10)]

        result = await svc.translate_payloads(payloads)

        assert result["translated_elements"][0]["text_content"] == "Tieu de"
        translator.translate_title.assert_awaited_once()
