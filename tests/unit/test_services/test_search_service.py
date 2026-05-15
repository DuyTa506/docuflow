"""
Unit tests for SearchService pagination correctness.

RED phase: test_page2_returns_different_items fails until sub-query .limit() is removed.
"""
import pytest
from unittest.mock import MagicMock, patch

from services.search_service import SearchService


def _make_doc(doc_id, title):
    d = MagicMock()
    d.id = doc_id
    d.title = title
    return d


class TestSearchPagination:
    def test_page2_returns_different_items(self):
        """page=2 must slice into results beyond the first page."""
        svc = SearchService()

        # 8 matching docs total
        docs = [_make_doc(f"DOC_{i:03d}", f"machine learning paper {i}") for i in range(8)]

        db = MagicMock()
        db.query.return_value.filter.return_value.ilike = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = docs
        db.query.return_value.filter.return_value.limit.return_value.all.return_value = docs

        r1 = svc.search(db, "machine", search_in=["title"], limit=3, offset=0)
        r2 = svc.search(db, "machine", search_in=["title"], limit=3, offset=3)

        ids_p1 = [r["document_id"] for r in r1["results"]]
        ids_p2 = [r["document_id"] for r in r2["results"]]

        assert r1["total"] == 8
        assert r2["total"] == 8
        assert len(ids_p1) == 3
        assert len(ids_p2) == 3
        assert ids_p1 != ids_p2, "page 1 and page 2 returned the same items — pagination bug"

    def test_last_page_returns_remainder(self):
        """page 3 of 8 items with limit=3 should return 2 items."""
        svc = SearchService()
        docs = [_make_doc(f"DOC_{i:03d}", f"title {i}") for i in range(8)]

        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = docs
        db.query.return_value.filter.return_value.limit.return_value.all.return_value = docs

        r = svc.search(db, "title", search_in=["title"], limit=3, offset=6)
        assert r["total"] == 8
        assert len(r["results"]) == 2
