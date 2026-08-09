"""SSE events endpoint: token auth via query param (EventSource can't set
headers) and per-user event visibility (non-admins only see their own
documents' task events).
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from serving.routers.events_router import _authenticate, _event_visible_to, _pg_dsn


class TestAuth:
    def test_valid_token_returns_payload(self):
        with patch(
            "services.auth_service.AuthService.decode_token",
            return_value={"sub": "USR_001", "role": "MEMBER"},
        ):
            payload = _authenticate("tok")
        assert payload["sub"] == "USR_001"

    def test_invalid_token_401(self):
        with patch("services.auth_service.AuthService.decode_token", return_value=None):
            with pytest.raises(HTTPException) as exc:
                _authenticate("bad")
        assert exc.value.status_code == 401


class TestVisibility:
    def test_admin_sees_everything(self):
        assert _event_visible_to({"document_id": "DOC_9"}, "USR_1", "ADMIN", {})

    def test_owner_sees_own_document(self):
        cache = {"DOC_9": "USR_1"}
        assert _event_visible_to({"document_id": "DOC_9"}, "USR_1", "MEMBER", cache)

    def test_non_owner_blocked(self):
        cache = {"DOC_9": "USR_2"}
        assert not _event_visible_to({"document_id": "DOC_9"}, "USR_1", "MEMBER", cache)

    def test_owner_lookup_cached_from_db(self):
        fake_doc = MagicMock()
        fake_doc.user_id = "USR_1"
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.query.return_value.filter.return_value.first.return_value = fake_doc

        cache: dict = {}
        with patch("data.database.get_db_manager") as mock_dbm:
            mock_dbm.return_value.session.return_value = session
            assert _event_visible_to({"document_id": "DOC_5"}, "USR_1", "MEMBER", cache)
        assert cache == {"DOC_5": "USR_1"}
        # second call hits the cache — no new session
        assert _event_visible_to({"document_id": "DOC_5"}, "USR_1", "MEMBER", cache)


def test_pg_dsn_strips_driver_suffix():
    with patch("serving.routers.events_router.settings") as mock_settings:
        mock_settings.database_url = "postgresql+psycopg2://u:p@h:5433/db"
        assert _pg_dsn() == "postgresql://u:p@h:5433/db"
