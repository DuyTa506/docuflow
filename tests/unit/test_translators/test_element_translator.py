"""Tests for ElementTranslator structure preservation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.translators.element_translator import ElementTranslator
from utils.translation_elements import layout_element_to_dict


def _elem(label, text, page_num=1, order=0):
    page = MagicMock()
    page.page_number = page_num
    elem = MagicMock()
    elem.label = label
    elem.text_content = text
    elem.sequence_order = order
    elem.bbox_x1 = 10
    elem.bbox_y1 = 20
    elem.bbox_x2 = 100
    elem.bbox_y2 = 50
    elem.page = page
    return elem


class TestElementTranslator:
    @pytest.mark.asyncio
    async def test_preserves_bbox_and_translates_text(self):
        translator = MagicMock()
        translator.translate_text = AsyncMock(return_value="Xin chao")
        translator.translate_title = AsyncMock(return_value="Tieu de")

        svc = ElementTranslator(translator)
        elements = [
            _elem("title", "Hello", order=0),
            _elem("text", "World", order=1),
            _elem("image", "ignored", order=2),
        ]
        payloads = [
            layout_element_to_dict(elem, elem.page.page_number if elem.page else 1)
            for elem in elements
        ]

        result = await svc.translate_payloads(payloads)

        assert result["translation_mode"] == "element_based"
        assert len(result["translated_elements"]) == 3
        assert result["translated_elements"][0]["text_content"] == "Tieu de"
        assert result["translated_elements"][0]["bbox"]["x1"] == 10
        assert result["translated_elements"][1]["text_content"] == "Xin chao"
        assert result["translated_elements"][2]["text_content"] == "ignored"
        assert "Tieu de" in result["translated_content"]
        assert "Xin chao" in result["translated_content"]
