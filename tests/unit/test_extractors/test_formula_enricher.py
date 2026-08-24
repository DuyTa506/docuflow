"""Routing and fallback tests for DeepSeek formula enrichment."""

import base64
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from core.models import ServicePageResult, UnifiedElement
from services.extractors.formula_enricher import (
    DeepSeekFormulaEnricher,
    is_complex_formula_page,
    merge_formula_markdown,
)


def _element(
    text: str = "$$x$$",
    *,
    element_type: str = "equation",
    order: int = 0,
    bbox: dict | None = None,
) -> UnifiedElement:
    return UnifiedElement(
        element_type=element_type,
        text=text,
        page_number=1,
        order=order,
        source="docling_layout",
        bbox=bbox or {"x1": 10, "y1": 20, "x2": 100, "y2": 40},
    )


def _image_b64(width: int = 200, height: int = 400) -> str:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_complex_router_uses_count_length_and_height():
    short = _element()
    assert not is_complex_formula_page([short], 800)
    assert is_complex_formula_page([short, _element(order=1), _element(order=2)], 800)
    assert is_complex_formula_page([_element("x" * 250)], 800)
    assert is_complex_formula_page(
        [_element(bbox={"x1": 10, "y1": 20, "x2": 100, "y2": 116})],
        800,
    )


@pytest.mark.asyncio
async def test_simple_formula_uses_tight_crop_and_normalizes_latex():
    target = _element()
    result = ServicePageResult(
        page_num=1,
        markdown=r"\[\frac{a}{b}\]",
        layout_elements=[
            {
                "label": "formula",
                "text_full": r"\[\frac{a}{b}\]",
                "x1": 0,
                "y1": 0,
                "x2": 100,
                "y2": 50,
            }
        ],
    )
    enricher = DeepSeekFormulaEnricher(object(), "/tmp/book.pdf")
    enricher._extract = AsyncMock(return_value=([], result))

    with patch(
        "services.extractors.formula_enricher._render_crop",
        return_value="/tmp/formula-test-missing.png",
    ):
        enriched = await enricher.enrich_page([target], 1, 600, 800)

    assert enriched[0].source == "deepseek_formula"
    assert enriched[0].element_type == "equation"
    assert enriched[0].text == r"$$\frac{a}{b}$$"
    enricher._extract.assert_awaited_once_with("/tmp/formula-test-missing.png", 1)


@pytest.mark.asyncio
async def test_code_misdetected_as_formula_becomes_text():
    target = _element()
    result = ServicePageResult(
        page_num=1,
        markdown="```java\nfor (int i=0; i<n; i++)\n```",
        layout_elements=[
            {
                "label": "text",
                "text_full": "```java\nfor (int i=0; i<n; i++)\n```",
                "x1": 0,
                "y1": 0,
                "x2": 100,
                "y2": 50,
            }
        ],
    )
    enricher = DeepSeekFormulaEnricher(object(), "/tmp/book.pdf")
    enricher._extract = AsyncMock(return_value=([], result))

    with patch(
        "services.extractors.formula_enricher._render_crop",
        return_value="/tmp/formula-test-missing.png",
    ):
        enriched = await enricher.enrich_page([target], 1, 600, 800)

    assert enriched[0].element_type == "text"
    assert enriched[0].source == "deepseek_formula"
    assert enriched[0].text.startswith("```java")


@pytest.mark.asyncio
async def test_complex_page_replaces_docling_formulas_with_full_page_layout():
    elements = [
        _element("intro", element_type="text", order=0),
        _element("$$bad 1$$", order=1),
        _element("$$bad 2$$", order=2),
        _element("$$bad 3$$", order=3),
    ]
    result = ServicePageResult(
        page_num=1,
        markdown="",
        image_base64=_image_b64(200, 400),
        layout_elements=[
            {
                "label": "formula",
                "text_full": r"\[a+b=c\]",
                "x1": 20,
                "y1": 40,
                "x2": 180,
                "y2": 80,
            },
            {
                "label": "text",
                "text_full": r"ordinary prose with inline \(x\)",
                "x1": 20,
                "y1": 100,
                "x2": 180,
                "y2": 120,
            },
        ],
    )
    enricher = DeepSeekFormulaEnricher(object(), "/tmp/book.pdf")
    enricher._extract = AsyncMock(return_value=([], result))

    enriched = await enricher.enrich_page(elements, 1, 600, 800)

    formulas = [element for element in enriched if element.element_type == "equation"]
    assert len(formulas) == 1
    assert formulas[0].text == "$$a+b=c$$"
    assert formulas[0].bbox == {"x1": 60.0, "y1": 80.0, "x2": 540.0, "y2": 160.0}
    assert r"ordinary prose with inline \(x\)" not in [element.text for element in enriched]
    enricher._extract.assert_awaited_once_with("/tmp/book.pdf", 1)


@pytest.mark.asyncio
async def test_deepseek_error_keeps_docling_formula():
    target = _element("$$raw PDF text$$")
    enricher = DeepSeekFormulaEnricher(object(), "/tmp/book.pdf")
    enricher._extract = AsyncMock(side_effect=RuntimeError("vLLM unavailable"))

    with patch(
        "services.extractors.formula_enricher._render_crop",
        return_value="/tmp/formula-test-missing.png",
    ):
        enriched = await enricher.enrich_page([target], 1, 600, 800)

    assert enriched == [target]
    assert enriched[0].source == "docling_layout"


def test_markdown_replaces_formula_without_reformatting_prose():
    original = [
        _element("Paragraph", element_type="text", order=0),
        _element("$$bad$$", order=1),
    ]
    enriched = [
        original[0],
        _element("$$good$$", order=1),
    ]
    enriched[1].source = "deepseek_formula"

    merged = merge_formula_markdown("Paragraph\n\n$$bad$$", original, enriched)

    assert merged == "Paragraph\n\n$$good$$"
