"""Khối "Thông tin quản trị CSDL" could never be filled in.

`DigestRenderer.render` accepted `reviewer` / `reviewer_approved` /
`entry_date`, but `export_service.build_digest_export` called `render(digest)`
with no arguments and nothing persisted them, so the last block of every
exported digest was three empty labels.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.digest_service import DigestService
from utils.digest_admin import DIGEST_ADMIN_KEYS, normalize_digest_admin


class TestNormalize:
    def test_keeps_only_the_three_known_fields(self):
        cleaned = normalize_digest_admin(
            {"reviewer": "A", "entry_date": "01/01/2026", "hacker": "x"}
        )

        assert set(cleaned) == set(DIGEST_ADMIN_KEYS)
        assert cleaned["reviewer"] == "A"
        assert cleaned["reviewer_approved"] == ""

    def test_trims_and_tolerates_none(self):
        assert normalize_digest_admin(None) == {k: "" for k in DIGEST_ADMIN_KEYS}
        assert normalize_digest_admin({"reviewer": "  A  "})["reviewer"] == "A"

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError):
            normalize_digest_admin(["A"])

    def test_rejects_non_string_value(self):
        with pytest.raises(ValueError):
            normalize_digest_admin({"reviewer": 42})


class TestAssemblePopulatesTheBlock:
    def _assemble(self, digest_admin):
        doc = SimpleNamespace(
            id="DOC_001",
            title="T",
            source_language="vi",
            original_filename="t.pdf",
            total_pages=10,
            bibliographic_metadata=None,
            usage_scope=None,
            digest_admin=digest_admin,
        )
        db = MagicMock()
        query = db.query.return_value.filter.return_value
        query.first.return_value = doc
        query.order_by.return_value.first.return_value = None
        query.order_by.return_value.limit.return_value.all.return_value = []
        db.query.return_value.join.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = (
            []
        )

        return DigestService().assemble(db, "DOC_001")

    def test_stored_values_reach_the_digest(self):
        digest = self._assemble(
            {
                "reviewer": "Nguyễn Văn A",
                "reviewer_approved": "Trần Văn B",
                "entry_date": "28/07/2026",
            }
        )

        assert digest.reviewer == "Nguyễn Văn A"
        assert digest.reviewer_approved == "Trần Văn B"
        assert digest.entry_date == "28/07/2026"

    def test_absent_column_value_is_not_an_error(self):
        digest = self._assemble(None)

        assert digest.reviewer == ""
