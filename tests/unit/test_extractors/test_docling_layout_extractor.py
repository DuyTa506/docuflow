"""Tests for Docling layout-preserving PDF extraction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.models import UnifiedElement
from services.extractors.docling_layout_extractor import (
    DoclingLayoutExtractor,
    _build_caption_map,
    _clean_figure_text,
    _figure_display_text,
    _formula_text,
    _is_tiny_picture,
    _map_element_type,
    prov_bbox_to_top_left,
)

PDF_PATH = Path(__file__).resolve().parents[3] / "2511.19575v2.pdf"


class TestConvertPipelineOptions:
    def test_images_scale_passed_from_settings(self):
        """Regression: confirmed live that Docling's own images_scale default
        (1.0, ~72dpi-equivalent) produces visibly blurry figure/chart crops
        (405x354px for a chart with data labels) once embedded in DOCX/PDF
        exports -- settings.docling_images_scale must reach PdfPipelineOptions."""
        from unittest.mock import MagicMock, patch

        ext = DoclingLayoutExtractor("/fake/path.pdf")

        with (
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.document_converter.DocumentConverter") as mock_converter_cls,
            patch("docling.document_converter.PdfFormatOption"),
        ):
            mock_converter_cls.return_value.convert.return_value = MagicMock(
                document=MagicMock(pages={})
            )
            with patch.object(ext, "_build_page_cache"):
                ext.convert()

        _, kwargs = mock_opts.call_args
        from config.settings import settings

        assert kwargs["images_scale"] == settings.docling_images_scale
        accelerator = kwargs["accelerator_options"]
        assert str(accelerator.device) == settings.docling_device
        assert accelerator.num_threads == settings.docling_num_threads
        table_options = kwargs["table_structure_options"]
        assert table_options.mode.value == settings.docling_table_mode
        assert kwargs["do_formula_enrichment"] is False


class TestBboxHelpers:
    def test_prov_bbox_to_top_left(self):
        bbox = SimpleNamespace(l=10.0, t=800.0, r=100.0, b=780.0)
        out = prov_bbox_to_top_left(bbox, page_height=841.89)
        assert out["x1"] == 10.0
        assert out["x2"] == 100.0
        assert out["y1"] == pytest.approx(41.89, abs=0.01)
        assert out["y2"] == pytest.approx(61.89, abs=0.01)

    def test_map_element_type(self):
        assert _map_element_type("table", None) == "table"
        assert _map_element_type("picture", None) == "figure"
        assert _map_element_type("formula", None) == "equation"
        assert _map_element_type("section_header", 2) == "heading"
        assert _map_element_type("text", None) == "text"


class TestFigureHelpers:
    def test_clean_figure_text_strips_html_and_markdown_image(self):
        raw = (
            "Figure 1: Chart\n\n"
            "<!-- Image not available -->\n"
            "![Image](data:image/png;base64,abc)"
        )
        assert _clean_figure_text(raw) == "Figure 1: Chart"

    def test_figure_display_text_prefers_caption(self):
        assert _figure_display_text("Figure 2: Architecture", "noise") == "Figure 2: Architecture"

    def test_figure_display_text_placeholder_when_empty(self):
        assert _figure_display_text("", "<!-- missing -->") == "(img_content)[figure]"

    def test_is_tiny_picture(self):
        assert _is_tiny_picture(30, 20, 40) is True
        assert _is_tiny_picture(405, 354, 40) is False


class TestCaptionAndFormulaHelpers:
    def test_build_caption_map(self):
        caption = SimpleNamespace(
            label=SimpleNamespace(value="caption"),
            parent=SimpleNamespace(cref="#/pictures/1"),
            text="Figure 1: Example.",
            prov=[SimpleNamespace(page_no=1, bbox=SimpleNamespace(l=0, t=0, r=1, b=1))],
        )

        class FakeDoc:
            def iterate_items(self):
                yield caption, 0

            def export_to_markdown(self, _item):
                return ""

        assert _build_caption_map(FakeDoc()) == {"#/pictures/1": "Figure 1: Example."}

    def test_formula_text_uses_orig_when_text_empty(self):
        item = SimpleNamespace(text="", latex=None, orig="L = α + β")
        assert _formula_text(item, None) == "L = α + β"

    def test_figure_unified_element_maps_to_figure_label(self):
        elem = UnifiedElement(
            element_type="figure",
            text="Figure 1: Chart",
            page_number=1,
            order=0,
            source="docling_layout",
            image_bytes_b64="abc",
        )
        assert elem.to_layout_element_dict()["label"] == "figure"


@pytest.mark.slow
@pytest.mark.skipif(not PDF_PATH.is_file(), reason="2511.19575v2.pdf not in repo root")
class TestDoclingLayoutExtractorIntegration:
    @pytest.fixture(scope="class")
    def extractor(self):
        ext = DoclingLayoutExtractor(str(PDF_PATH))
        ext.convert()
        return ext

    def test_page1_has_title_not_chart_fragments(self, extractor):
        md = extractor.page_markdown(1)
        assert "HunyuanOCR Technical Report" in md
        assert "Abstract" in md
        assert "\n55\n" not in md

    def test_page2_starts_with_proper_flow(self, extractor):
        md = extractor.page_markdown(2)
        assert "Introduction" in md
        assert not md.lstrip().startswith("data and, for the first time")

    def test_page3_has_table_structure(self, extractor):
        md = extractor.page_markdown(3)
        assert "|" in md
        assert "Model Type" in md or "Model Name" in md
        elements = extractor.extract_page(3)
        assert any(e.element_type == "table" for e in elements)

    def test_reading_order_elements(self, extractor):
        p1 = extractor.extract_page(1)
        assert p1
        assert p1[0].source == "docling_layout"
        headings = [e for e in p1 if e.element_type == "heading"]
        assert any("HunyuanOCR" in e.text for e in headings)

    def test_figures_extracted_with_images_and_captions(self, extractor):
        all_elems = []
        for pn in range(1, extractor.total_pages + 1):
            all_elems.extend(extractor.extract_page(pn))
        figures = [e for e in all_elems if e.element_type == "figure"]
        assert len(figures) >= 30
        with_pixels = [e for e in figures if e.image_bytes_b64]
        assert len(with_pixels) >= 25
        with_caption = [e for e in figures if e.text.startswith("Figure")]
        assert len(with_caption) >= 3

    def test_formulas_extracted(self, extractor):
        all_elems = []
        for pn in range(1, extractor.total_pages + 1):
            all_elems.extend(extractor.extract_page(pn))
        equations = [e for e in all_elems if e.element_type == "equation"]
        assert len(equations) >= 1
        assert equations[0].text.startswith("$$")

    def test_page1_main_figure_has_caption_and_image(self, extractor):
        figures = [e for e in extractor.extract_page(1) if e.element_type == "figure"]
        assert figures
        main = max(figures, key=lambda e: (e.image_width or 0) * (e.image_height or 0))
        assert "Figure 1" in main.text
        assert main.image_bytes_b64
        assert (main.image_width or 0) > 200
