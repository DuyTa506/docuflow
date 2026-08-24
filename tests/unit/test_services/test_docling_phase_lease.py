"""Docling model ownership spans the complete text-page phase."""

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_one_lease_wraps_all_docling_slices():
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

    @asynccontextmanager
    async def fake_lease(*args, **kwargs):
        lease_calls.append(kwargs)
        events.append(("lease", "enter"))
        yield
        events.append(("lease", "exit"))

    saved = []
    with (
        patch("services.gpu_lease.gpu_lease", fake_lease),
        patch("utils.image_utils.render_pdf_page_to_base64", return_value="image"),
    ):
        await DocumentService()._extract_text_pages(
            file_path="/tmp/fake.pdf",
            pages=list(range(1, 12)),  # two DOCLING_PAGE_CHUNK windows
            extractor=FakeExtractor(),
            save_page=lambda page, **kwargs: saved.append(page),
            on_page_done=lambda: None,
            lease_key="docling-text:DOC_TEST",
        )

    assert events[0] == ("lease", "enter")
    assert [event for event in events if event[0] == "convert"] == [
        ("convert", (1, 10)),
        ("convert", (11, 11)),
    ]
    assert events[-1] == ("lease", "exit")
    assert saved == list(range(1, 12))
    assert lease_calls[0]["slots"] == 4
