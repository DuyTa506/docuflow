from unittest.mock import patch, MagicMock


def _extraction(eid="KW_EXT_001"):
    e = MagicMock()
    e.id = eid
    e.document_id = "DOC_001"
    e.status = "COMPLETED"
    e.max_keywords = 20
    e.total_keywords = 5
    e.error = None
    e.created_at = None
    e.updated_at = None
    return e


class TestStartKeywordExtraction:
    def test_success(self, client):
        with patch("serving.routers.keywords_router._svc") as mock_svc:
            mock_svc.submit.return_value = ("TASK_001", "KW_EXT_001", False)
            resp = client.post("/api/v2/documents/DOC_001/keywords", json={"max_keywords": 20})
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "TASK_001"
        assert resp.json()["resource_id"] == "KW_EXT_001"

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.keywords_router._svc") as mock_svc:
            mock_svc.submit.side_effect = ValueError("Document not found")
            resp = client.post("/api/v2/documents/DOC_999/keywords")
        assert resp.status_code == 404


class TestGetKeywords:
    def test_success(self, client):
        mock_ext = _extraction()
        with patch("serving.routers.keywords_router.DocumentRepository") as MockDocRepo, \
             patch("serving.routers.keywords_router.KeywordRepository") as MockKwRepo:
            MockDocRepo.return_value.get.return_value = MagicMock()
            MockKwRepo.return_value.get_for_document.return_value = []
            MockKwRepo.return_value.get_latest_extraction.return_value = mock_ext
            resp = client.get("/api/v2/documents/DOC_001/keywords")
        assert resp.status_code == 200
        assert resp.json()["document_id"] == "DOC_001"
        assert resp.json()["keywords"] == []

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.keywords_router.DocumentRepository") as MockDocRepo:
            MockDocRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_999/keywords")
        assert resp.status_code == 404


class TestListKeywordExtractions:
    def test_success(self, client):
        mock_ext = _extraction()
        with patch("serving.routers.keywords_router.DocumentRepository") as MockDocRepo, \
             patch("serving.routers.keywords_router.KeywordRepository") as MockKwRepo:
            MockDocRepo.return_value.get.return_value = MagicMock()
            MockKwRepo.return_value.list_extractions.return_value = [mock_ext]
            resp = client.get("/api/v2/documents/DOC_001/keywords/extractions")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["id"] == "KW_EXT_001"

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.keywords_router.DocumentRepository") as MockDocRepo:
            MockDocRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_999/keywords/extractions")
        assert resp.status_code == 404


class TestGetKeywordExtraction:
    def test_success(self, client):
        mock_ext = _extraction()
        with patch("serving.routers.keywords_router.KeywordRepository") as MockRepo:
            MockRepo.return_value.get_extraction.return_value = mock_ext
            resp = client.get("/api/v2/documents/DOC_001/keywords/extractions/KW_EXT_001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "KW_EXT_001"
        assert resp.json()["status"] == "COMPLETED"

    def test_not_found_returns_404(self, client):
        with patch("serving.routers.keywords_router.KeywordRepository") as MockRepo:
            MockRepo.return_value.get_extraction.return_value = None
            resp = client.get("/api/v2/documents/DOC_001/keywords/extractions/KW_EXT_999")
        assert resp.status_code == 404
