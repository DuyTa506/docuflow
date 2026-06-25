from unittest.mock import patch, MagicMock


def _extraction(eid="RES_EXT_001"):
    e = MagicMock()
    e.id = eid
    e.document_id = "DOC_001"
    e.status = "COMPLETED"
    e.total_directions = 3
    e.error = None
    e.created_at = None
    e.updated_at = None
    return e


def _catalog_dir(did="RD_001"):
    rd = MagicMock()
    rd.id = did
    rd.direction_name = "Machine Learning"
    rd.is_predefined = True
    rd.created_at = None
    return rd


class TestStartResearchExtraction:
    def test_success(self, client):
        with patch("serving.routers.research_router._svc") as mock_svc:
            mock_svc.submit.return_value = ("TASK_001", "RES_EXT_001", False)
            resp = client.post("/api/v2/documents/DOC_001/research-directions")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "TASK_001"

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.research_router._svc") as mock_svc:
            mock_svc.submit.side_effect = ValueError("Document not found")
            resp = client.post("/api/v2/documents/DOC_999/research-directions")
        assert resp.status_code == 404


class TestGetResearchDirections:
    def test_success(self, client):
        mock_ext = _extraction()
        with patch("serving.routers.research_router.DocumentRepository") as MockDocRepo, \
             patch("serving.routers.research_router.ResearchRepository") as MockResRepo:
            MockDocRepo.return_value.get.return_value = MagicMock()
            MockResRepo.return_value.get_directions.return_value = []
            MockResRepo.return_value.get_latest_extraction.return_value = mock_ext
            resp = client.get("/api/v2/documents/DOC_001/research-directions")
        assert resp.status_code == 200
        assert resp.json()["document_id"] == "DOC_001"
        assert resp.json()["directions"] == []

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.research_router.DocumentRepository") as MockDocRepo:
            MockDocRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_999/research-directions")
        assert resp.status_code == 404


class TestListResearchExtractions:
    def test_success(self, client):
        mock_ext = _extraction()
        with patch("serving.routers.research_router.DocumentRepository") as MockDocRepo, \
             patch("serving.routers.research_router.ResearchRepository") as MockResRepo:
            MockDocRepo.return_value.get.return_value = MagicMock()
            MockResRepo.return_value.list_extractions.return_value = [mock_ext]
            resp = client.get("/api/v2/documents/DOC_001/research-directions/extractions")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.research_router.DocumentRepository") as MockDocRepo:
            MockDocRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_999/research-directions/extractions")
        assert resp.status_code == 404


class TestGetResearchExtraction:
    def test_success(self, client):
        mock_ext = _extraction()
        with patch("serving.routers.research_router.ResearchRepository") as MockRepo:
            MockRepo.return_value.get_extraction.return_value = mock_ext
            resp = client.get("/api/v2/documents/DOC_001/research-directions/extractions/RES_EXT_001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "RES_EXT_001"

    def test_not_found_returns_404(self, client):
        with patch("serving.routers.research_router.ResearchRepository") as MockRepo:
            MockRepo.return_value.get_extraction.return_value = None
            resp = client.get("/api/v2/documents/DOC_001/research-directions/extractions/RES_999")
        assert resp.status_code == 404


class TestAddCatalogDirection:
    def test_success_returns_201(self, admin_client):
        mock_rd = _catalog_dir()
        with patch("serving.routers.research_router.ResearchRepository") as MockRepo:
            MockRepo.return_value.get_by_name.return_value = None
            MockRepo.return_value.add_catalog.return_value = mock_rd
            resp = admin_client.post(
                "/api/v2/research-directions/catalog",
                json={"direction_name": "Machine Learning"},
            )
        assert resp.status_code == 201
        assert resp.json()["direction_name"] == "Machine Learning"

    def test_duplicate_returns_400(self, admin_client):
        with patch("serving.routers.research_router.ResearchRepository") as MockRepo:
            MockRepo.return_value.get_by_name.return_value = _catalog_dir()
            resp = admin_client.post(
                "/api/v2/research-directions/catalog",
                json={"direction_name": "Machine Learning"},
            )
        assert resp.status_code == 400

    def test_member_cannot_add_returns_403(self, client):
        resp = client.post(
            "/api/v2/research-directions/catalog",
            json={"direction_name": "Machine Learning"},
        )
        assert resp.status_code == 403


class TestListCatalog:
    def test_success(self, client):
        mock_rd = _catalog_dir()
        with patch("serving.routers.research_router.ResearchRepository") as MockRepo:
            MockRepo.return_value.get_catalog.return_value = [mock_rd]
            resp = client.get("/api/v2/research-directions/catalog")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["direction_name"] == "Machine Learning"
