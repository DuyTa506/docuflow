from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def _summary(sid="SUM_001"):
    s = MagicMock()
    s.id = sid
    s.document_id = "DOC_001"
    s.summary_type = "short"
    s.content = "Summary content"
    s.status = "COMPLETED"
    s.created_at = None
    s.updated_at = None
    return s


class TestStartSummarization:
    def test_success(self, client):
        with patch("serving.routers.summarization_router._svc") as mock_svc:
            mock_svc.submit.return_value = ("TASK_001", "SUM_001", False)
            resp = client.post(
                "/api/v2/documents/DOC_001/summaries", json={"summary_type": "short"}
            )
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "TASK_001"
        assert resp.json()["resource_id"] == "SUM_001"

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.summarization_router._svc") as mock_svc:
            mock_svc.submit.side_effect = ValueError("Document not found")
            resp = client.post("/api/v2/documents/DOC_999/summaries")
        assert resp.status_code == 404


class TestListSummaries:
    def test_success(self, client):
        mock_s = _summary()
        with (
            patch("serving.routers.summarization_router.DocumentRepository") as MockDocRepo,
            patch("serving.routers.summarization_router.SummaryRepository") as MockSumRepo,
        ):
            MockDocRepo.return_value.get.return_value = MagicMock()
            MockSumRepo.return_value.list.return_value = [mock_s]
            resp = client.get("/api/v2/documents/DOC_001/summaries")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["summary_type"] == "short"

    def test_document_not_found_returns_404(self, client):
        with patch(
            "serving.routers.summarization_router.get_authorized_document",
            side_effect=HTTPException(status_code=404, detail="Document not found"),
        ):
            resp = client.get("/api/v2/documents/DOC_999/summaries")
        assert resp.status_code == 404


class TestGetSummary:
    def test_success(self, client):
        mock_s = _summary()
        with patch("serving.routers.summarization_router.SummaryRepository") as MockRepo:
            MockRepo.return_value.get.return_value = mock_s
            resp = client.get("/api/v2/documents/DOC_001/summaries/SUM_001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "SUM_001"
        assert resp.json()["content"] == "Summary content"

    def test_not_found_returns_404(self, client):
        with patch("serving.routers.summarization_router.SummaryRepository") as MockRepo:
            MockRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_001/summaries/SUM_999")
        assert resp.status_code == 404


class TestUploadSummary:
    def test_success(self, client):
        mock_s = _summary()
        with (
            patch("serving.routers.summarization_router.SummaryRepository") as MockRepo,
            patch(
                "serving.routers.summarization_router.extract_text_from_upload",
                new=AsyncMock(return_value="Corrected summary"),
            ),
        ):
            MockRepo.return_value.update.return_value = mock_s
            resp = client.post(
                "/api/v2/documents/DOC_001/summaries/SUM_001/upload",
                files={"file": ("fix.txt", b"Corrected summary", "text/plain")},
            )
        assert resp.status_code == 200
        assert resp.json()["id"] == "SUM_001"

    def test_not_found_returns_404(self, client):
        with (
            patch("serving.routers.summarization_router.SummaryRepository") as MockRepo,
            patch(
                "serving.routers.summarization_router.extract_text_from_upload",
                new=AsyncMock(return_value="text"),
            ),
        ):
            MockRepo.return_value.update.return_value = None
            resp = client.post(
                "/api/v2/documents/DOC_001/summaries/SUM_999/upload",
                files={"file": ("fix.txt", b"text", "text/plain")},
            )
        assert resp.status_code == 404
