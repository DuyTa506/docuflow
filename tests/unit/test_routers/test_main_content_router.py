from unittest.mock import MagicMock, patch


def _mc(mc_id="MC_001"):
    mc = MagicMock()
    mc.id = mc_id
    mc.document_id = "DOC_001"
    mc.details = {"sections": []}
    mc.status = "COMPLETED"
    mc.created_at = None
    mc.updated_at = None
    return mc


class TestStartMainContentExtraction:
    def test_success(self, client):
        with patch("serving.routers.main_content_router._svc") as mock_svc:
            mock_svc.submit.return_value = ("TASK_001", "MC_001", False)
            resp = client.post("/api/v2/documents/DOC_001/main-content")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "TASK_001"
        assert resp.json()["resource_id"] == "MC_001"

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.main_content_router._svc") as mock_svc:
            mock_svc.submit.side_effect = ValueError("Document not found")
            resp = client.post("/api/v2/documents/DOC_999/main-content")
        assert resp.status_code == 404


class TestGetMainContent:
    def test_success(self, client):
        mock_mc = _mc()
        with (
            patch("serving.routers.main_content_router.DocumentRepository") as MockDocRepo,
            patch("serving.routers.main_content_router.MainContentRepository") as MockMCRepo,
        ):
            MockDocRepo.return_value.get.return_value = MagicMock()
            MockMCRepo.return_value.list.return_value = [mock_mc]
            resp = client.get("/api/v2/documents/DOC_001/main-content")
        assert resp.status_code == 200
        assert resp.json()["id"] == "MC_001"

    def test_no_content_yet_returns_message(self, client):
        with (
            patch("serving.routers.main_content_router.DocumentRepository") as MockDocRepo,
            patch("serving.routers.main_content_router.MainContentRepository") as MockMCRepo,
        ):
            MockDocRepo.return_value.get.return_value = MagicMock()
            MockMCRepo.return_value.list.return_value = []
            resp = client.get("/api/v2/documents/DOC_001/main-content")
        assert resp.status_code == 200
        assert "message" in resp.json()

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.main_content_router.DocumentRepository") as MockDocRepo:
            MockDocRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_999/main-content")
        assert resp.status_code == 404


class TestListMainContent:
    def test_success(self, client):
        mock_mc = _mc()
        with (
            patch("serving.routers.main_content_router.DocumentRepository") as MockDocRepo,
            patch("serving.routers.main_content_router.MainContentRepository") as MockMCRepo,
        ):
            MockDocRepo.return_value.get.return_value = MagicMock()
            MockMCRepo.return_value.list.return_value = [mock_mc]
            resp = client.get("/api/v2/documents/DOC_001/main-content/list")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.main_content_router.DocumentRepository") as MockDocRepo:
            MockDocRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_999/main-content/list")
        assert resp.status_code == 404


class TestGetMainContentById:
    def test_success(self, client):
        mock_mc = _mc()
        with patch("serving.routers.main_content_router.MainContentRepository") as MockRepo:
            MockRepo.return_value.get.return_value = mock_mc
            resp = client.get("/api/v2/documents/DOC_001/main-content/MC_001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "MC_001"
        assert resp.json()["status"] == "COMPLETED"

    def test_not_found_returns_404(self, client):
        with patch("serving.routers.main_content_router.MainContentRepository") as MockRepo:
            MockRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_001/main-content/MC_999")
        assert resp.status_code == 404
