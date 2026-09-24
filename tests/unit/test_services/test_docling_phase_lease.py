"""Docling lease covers convert+read only; formula OCR runs after release."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_lease_released_before_formula_enrichment():
    from services.document_service import DocumentService

    events = []
    lease_calls = []

    class FakeElement:
        def to_layout_element_dict(self):
            return {}

    class FakeExtractor:
        def convert(self, page_range=None):
            events.append(("convert", page_range))

        def extract_page(self, page_number):
            return [FakeElement()]

        def page_size(self, page_number):
            return (100, 200)

        def page_markdown(self, page_number):
            return f"page {page_number}"

    class FakeEnricher:
        async def enrich_page(self, elements, page_number, page_w, page_h):
            events.append(("enrich", page_number))
            return list(elements)

    @asynccontextmanager
    async def fake_lease(*args, **kwargs):
        lease_calls.append(kwargs)
        events.append(("lease", "enter"))
        yield SimpleNamespace(abort=SimpleNamespace(is_set=lambda: False))
        events.append(("lease", "exit"))

    saved = []
    with (
        patch("services.gpu_lease.gpu_lease", fake_lease),
        patch(
            "services.extractors.formula_enricher.merge_formula_markdown",
            side_effect=lambda md, _o, _e: md,
        ),
        patch("utils.image_utils.render_pdf_page_to_base64", return_value="image"),
    ):
        await DocumentService()._extract_text_pages(
            file_path="/tmp/fake.pdf",
            pages=list(range(1, 12)),  # two DOCLING_PAGE_CHUNK windows
            extractor=FakeExtractor(),
            save_page=lambda page, **kwargs: saved.append(page),
            on_page_done=lambda: None,
            formula_enricher=FakeEnricher(),
            lease_key="docling-text:DOC_TEST",
        )

    # Per slice: enter → convert → exit → enrich… (lease never held during OCR)
    assert events[0] == ("lease", "enter")
    assert ("convert", (1, 10)) in events
    assert ("convert", (11, 11)) in events
    first_exit = events.index(("lease", "exit"))
    first_enrich = next(i for i, e in enumerate(events) if e[0] == "enrich")
    assert first_exit < first_enrich
    assert saved == list(range(1, 12))
    assert len(lease_calls) == 2  # one lease per slice
    assert lease_calls[0]["slots"] == 4
