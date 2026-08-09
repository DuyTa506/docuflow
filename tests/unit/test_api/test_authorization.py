"""Cross-user authorization tests for document-scoped endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_current_user, get_db
from serving.workflow_api import app

pytest_plugins = ["tests.unit.test_routers.conftest"]


def _user(uid="USR_001", role="MEMBER"):
    u = MagicMock()
    u.id = uid
    u.role = role
    u.status = "ACTIVE"
    return u


def _doc(user_id="USR_OTHER"):
    d = MagicMock()
    d.id = "DOC_001"
    d.user_id = user_id
    d.title = "Other Doc"
    d.original_filename = "other.pdf"
    d.format = "pdf"
    d.file_path = "/tmp/other.pdf"
    return d


@pytest.fixture
def auth_client(mock_db):
    user = _user()
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app, raise_server_exceptions=True)
    app.dependency_overrides.clear()


class TestDocumentAuthorization:
    def test_extract_other_users_doc_returns_403(self, auth_client):
        with patch("serving.routers.documents_router.get_authorized_document") as mock_auth:
            from fastapi import HTTPException

            mock_auth.side_effect = HTTPException(status_code=403, detail="Access denied")
            resp = auth_client.post("/api/v2/documents/DOC_001/extract")
        assert resp.status_code == 403

    def test_digest_other_users_doc_returns_403(self, auth_client):
        with patch("serving.routers.digest_router.get_authorized_document") as mock_auth:
            from fastapi import HTTPException

            mock_auth.side_effect = HTTPException(status_code=403, detail="Access denied")
            resp = auth_client.post("/api/v2/documents/DOC_001/digest")
        assert resp.status_code == 403


class TestTaskAuthorization:
    def test_get_task_other_users_doc_returns_403(self, auth_client):
        mock_status = {
            "task_id": "TASK_001",
            "document_id": "DOC_001",
            "status": "RUNNING",
            "error": "secret traceback\nline2",
        }
        with (
            patch("serving.routers.tasks_router.task_manager") as mock_tm,
            patch("data.repositories.DocumentRepository") as MockRepo,
        ):
            mock_tm.get_status.return_value = mock_status
            MockRepo.return_value.get.return_value = _doc(user_id="USR_OTHER")
            resp = auth_client.get("/api/v2/tasks/TASK_001")
        assert resp.status_code == 403

    def test_get_task_error_sanitized_for_member(self, auth_client, mock_db):
        mock_status = {
            "task_id": "TASK_001",
            "document_id": "DOC_001",
            "status": "FAILED",
            "error": "Boom\nTraceback (most recent call last):\n  secret",
        }
        with (
            patch("serving.routers.tasks_router.task_manager") as mock_tm,
            patch("data.repositories.DocumentRepository") as MockRepo,
        ):
            mock_tm.get_status.return_value = mock_status
            MockRepo.return_value.get.return_value = _doc(user_id="USR_001")
            resp = auth_client.get("/api/v2/tasks/TASK_001")
        assert resp.status_code == 200
        assert "Traceback" not in resp.json().get("error", "")
