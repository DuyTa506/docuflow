"""Tests for bbox-based spatial DOCX rendering."""

import io

from docx import Document as DocxDocument
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

from utils.markdown_docx import render_layout_elements_to_docx
from utils.translation_elements import TranslatedElementView


class TestSpatialDocx:
    def test_center_alignment_from_bbox(self):
        doc = DocxDocument()
        elements = [
            TranslatedElementView(
                label="text",
                text_content="Hebertt Sira-Ramírez",
                page_number=1,
                bbox_x1=200,
                bbox_y1=100,
                bbox_x2=400,
                bbox_y2=120,
            ),
            TranslatedElementView(
                label="text",
                text_content="Control Design",
                page_number=1,
                bbox_x1=180,
                bbox_y1=130,
                bbox_x2=420,
                bbox_y2=150,
            ),
        ]
        render_layout_elements_to_docx(doc, elements)
        aligned = [
            p for p in doc.paragraphs
            if p.text and p.alignment == WD_PARAGRAPH_ALIGNMENT.CENTER
        ]
        assert len(aligned) >= 1

    def test_equation_element_renders(self):
        doc = DocxDocument()
        elements = [
            TranslatedElementView(
                label="formula",
                text_content="E = mc^2",
                page_number=1,
                bbox_x1=100,
                bbox_y1=50,
                bbox_x2=300,
                bbox_y2=70,
            ),
        ]
        render_layout_elements_to_docx(doc, elements)
        buf = io.BytesIO()
        doc.save(buf)
        assert len(buf.getvalue()) > 1000
