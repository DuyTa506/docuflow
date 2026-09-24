"""Tests for BlockTranslator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

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
            _payload(1, 2, "image", "(img_content)[image_1]", y1=100),
        ]

        result = await svc.translate_payloads(payloads)

        assert result["translation_mode"] == "block_based"
        assert len(result["translated_elements"]) == 2
        assert result["translated_elements"][0]["text_content"] == "Xin chao the gioi"
        assert result["translated_elements"][0]["bbox"]["y1"] == 10
        assert result["translated_elements"][1]["label"] == "image"
        assert result["translated_elements"][1]["text_content"] == "(img_content)[image_1]"
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


def _upper_cells():
    async def _cells(texts):
        return [t.upper() for t in texts]

    return AsyncMock(side_effect=_cells)


class TestTablesAndCaptionsAreTranslated:
    """Live regression (E2E Q2/Q3): every table stayed in the source language
    (Digital IC Design 46/46, Switching PSU 60/60) and Docling figure captions
    were skipped — both were passed through as images are."""

    @pytest.mark.asyncio
    async def test_html_table_cells_translated_tags_and_numbers_kept(self):
        translator = MagicMock()
        translator.translate_cells = _upper_cells()
        html = (
            '<table><tr><td colspan="2">Core loss</td><td>.650</td></tr>'
            "<tr><td>Part <b>number</b></td><td>50B10-5D</td><td>348,000</td></tr></table>"
        )

        result = await BlockTranslator(translator).translate_payloads(
            [_payload(1, 0, "table", html)]
        )

        out = result["translated_elements"][0]["text_content"]
        assert out == (
            '<table><tr><td colspan="2">CORE LOSS</td><td>.650</td></tr>'
            "<tr><td>PART <b>NUMBER</b></td><td>50B10-5D</td><td>348,000</td></tr></table>"
        )
        sent = translator.translate_cells.await_args[0][0]
        assert ".650" not in sent and "50B10-5D" not in sent

    @pytest.mark.asyncio
    async def test_markdown_pipe_table_cells_translated(self):
        translator = MagicMock()
        translator.translate_cells = _upper_cells()
        table = "| Name | Value |\n|---|---|\n| Voltage | 12 |"

        result = await BlockTranslator(translator).translate_payloads(
            [_payload(1, 0, "table", table)]
        )

        assert result["translated_elements"][0]["text_content"] == (
            "| NAME | VALUE |\n|---|---|\n| VOLTAGE | 12 |"
        )

    @pytest.mark.asyncio
    async def test_figure_caption_translated_image_kept(self):
        translator = MagicMock()
        translator.translate_title = AsyncMock(return_value="Hình 1. Sơ đồ khối")
        payload = _payload(1, 0, "figure", "Figure 1. Block diagram")
        payload["crop_image_key"] = "documents/D/crops/1.jpg"

        result = await BlockTranslator(translator).translate_payloads([payload])

        out = result["translated_elements"][0]
        assert out["text_content"] == "Hình 1. Sơ đồ khối"
        assert out["crop_image_key"] == "documents/D/crops/1.jpg"

    @pytest.mark.asyncio
    async def test_figure_placeholder_not_sent_to_llm(self):
        translator = MagicMock()
        translator.translate_title = AsyncMock()

        result = await BlockTranslator(translator).translate_payloads(
            [_payload(1, 0, "figure", "(img_content)[figure]")]
        )

        assert result["translated_elements"][0]["text_content"] == "(img_content)[figure]"
        translator.translate_title.assert_not_awaited()
