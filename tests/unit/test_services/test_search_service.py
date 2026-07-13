"""
Unit tests for SearchService pagination correctness.
"""

from unittest.mock import MagicMock, patch

import pytest

from services.search_service import SearchService


def _make_doc(doc_id, title, **kwargs):
    d = MagicMock()
    d.id = doc_id
    d.title = title
    d.original_filename = kwargs.get("original_filename", f"{title}.pdf")
    d.format = kwargs.get("format", "pdf")
    d.total_pages = kwargs.get("total_pages", 10)
    d.processing_status = kwargs.get("processing_status", "EXTRACTED")
    d.source_language = kwargs.get("source_language", "en")
    d.created_at = kwargs.get("created_at", None)
    return d


class TestSearchPagination:
    def test_page2_returns_different_items(self):
        """page=2 must slice into items beyond the first page."""
        svc = SearchService()
        docs = [_make_doc(f"DOC_{i:03d}", f"machine learning paper {i}") for i in range(8)]

        db = MagicMock()
        db.query.return_value.filter.return_value.ilike = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = docs
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        with patch.object(svc, "_visible_doc_ids", return_value=None):
            r1 = svc.search(db, "machine", search_in=["title"], limit=3, offset=0, is_admin=True)
            r2 = svc.search(db, "machine", search_in=["title"], limit=3, offset=3, is_admin=True)

        ids_p1 = [r["id"] for r in r1["items"]]
        ids_p2 = [r["id"] for r in r2["items"]]

        assert r1["total"] == 8
        assert r2["total"] == 8
        assert len(ids_p1) == 3
        assert len(ids_p2) == 3
        assert ids_p1 != ids_p2, "page 1 and page 2 returned the same items — pagination bug"

    def test_last_page_returns_remainder(self):
        """offset 6 of 8 items with limit=3 should return 2 items."""
        svc = SearchService()
        docs = [_make_doc(f"DOC_{i:03d}", f"title {i}") for i in range(8)]

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = docs
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        with patch.object(svc, "_visible_doc_ids", return_value=None):
            r = svc.search(db, "title", search_in=["title"], limit=3, offset=6, is_admin=True)

        assert r["total"] == 8
        assert len(r["items"]) == 2

    def test_items_use_document_list_shape(self):
        """Each hit must expose ``id`` (not document_id) plus list metadata."""
        svc = SearchService()
        doc = _make_doc("DOC_001", "Alpha paper")

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [doc]
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        with patch.object(svc, "_visible_doc_ids", return_value=None):
            r = svc.search(db, "Alpha", search_in=["title"], limit=5, offset=0, is_admin=True)

        assert len(r["items"]) == 1
        item = r["items"][0]
        assert item["id"] == "DOC_001"
        assert item["title"] == "Alpha paper"
        assert item["format"] == "pdf"
        assert item["processing_status"] == "EXTRACTED"
        assert item["match_field"] == "title"
        assert "document_id" not in item
