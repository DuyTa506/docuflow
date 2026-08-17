"""Digest tree freshness checks."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from utils.tree_quality import TREE_SCHEMA_VERSION


def test_tree_index_fresh_rejects_old_schema():
    from workflows.activities import digest_activities

    row = MagicMock()
    row.created_at = datetime.utcnow()
    row.config = {"tree_schema_version": 0, "tree_quality": {"ok": True}}

    with patch.object(digest_activities, "get_db_manager") as mock_db:
        session = MagicMock()
        mock_db.return_value.session.return_value.__enter__.return_value = session
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = row
        assert digest_activities._tree_index_fresh("DOC_1") is False


def test_tree_index_fresh_rejects_failed_quality():
    from workflows.activities import digest_activities

    row = MagicMock()
    row.created_at = datetime.utcnow()
    row.config = {"tree_schema_version": TREE_SCHEMA_VERSION, "tree_quality": {"ok": False}}

    with patch.object(digest_activities, "get_db_manager") as mock_db:
        session = MagicMock()
        mock_db.return_value.session.return_value.__enter__.return_value = session
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value = row
        assert digest_activities._tree_index_fresh("DOC_1") is False
