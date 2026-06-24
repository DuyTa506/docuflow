"""Tests for DocxInPlaceTranslator."""

import io
import pytest
from unittest.mock import AsyncMock, MagicMock

from docx import Document as DocxDocument

from services.translators.docx_inplace_translator import DocxInPlaceTranslator


@pytest.fixture
def sample_docx(tmp_path):
    path = tmp_path / "sample.docx"
    doc = DocxDocument()
    doc.add_heading("Hello", level=1)
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Cell A"
    table.rows[0].cells[1].text = "Cell B"
    doc.save(path)
    return path


class TestDocxInPlaceTranslator:
    @pytest.mark.asyncio
    async def test_translates_paragraphs_and_tables(self, sample_docx, tmp_path):
        out = tmp_path / "out.docx"
        translator = MagicMock()
        translator.translate_text = AsyncMock(side_effect=lambda t: f"VI:{t}")

        svc = DocxInPlaceTranslator(translator)
        result = await svc.translate_file(str(sample_docx), str(out), doc_format="docx")

        assert result["translation_mode"] == "docx_inplace"
        assert out.exists()

        doc = DocxDocument(str(out))
        texts = [p.text for p in doc.paragraphs if p.text.strip()]
        assert any("VI:Hello" in t for t in texts)

        table_text = " ".join(
            cell.text for table in doc.tables for row in table.rows for cell in row.cells
        )
        assert "VI:Cell A" in table_text
        assert "VI:Cell B" in table_text
