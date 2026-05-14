from unittest.mock import patch, MagicMock, AsyncMock


def _translation(tid="TRANS_001"):
    t = MagicMock()
    t.id = tid
    t.document_id = "DOC_001"
    t.target_language = "vi"
    t.translated_content = "Translated text"
    t.status = "COMPLETED"
    t.created_at = None
    t.updated_at = None
    return t


class TestStartTranslation:
    def test_success(self, client):
        with patch("serving.routers.translation_router._svc") as mock_svc:
            mock_svc.submit.return_value = ("TASK_001", "TRANS_001")
            resp = client.post("/api/v2/documents/DOC_001/translations", json={
                "target_language": "vi",
                "domain": "general",
            })
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "TASK_001"
        assert resp.json()["resource_id"] == "TRANS_001"

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.translation_router._svc") as mock_svc:
            mock_svc.submit.side_effect = ValueError("Document not found")
            resp = client.post("/api/v2/documents/DOC_999/translations")
        assert resp.status_code == 404


class TestListTranslations:
    def test_success(self, client):
        mock_t = _translation()
        with patch("serving.routers.translation_router.DocumentRepository") as MockDocRepo, \
             patch("serving.routers.translation_router.TranslationRepository") as MockTransRepo:
            MockDocRepo.return_value.get.return_value = MagicMock()
            MockTransRepo.return_value.list.return_value = [mock_t]
            resp = client.get("/api/v2/documents/DOC_001/translations")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["target_language"] == "vi"

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.translation_router.DocumentRepository") as MockDocRepo:
            MockDocRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_999/translations")
        assert resp.status_code == 404


class TestGetTranslation:
    def test_success(self, client):
        mock_t = _translation()
        with patch("serving.routers.translation_router.TranslationRepository") as MockRepo:
            MockRepo.return_value.get.return_value = mock_t
            resp = client.get("/api/v2/documents/DOC_001/translations/TRANS_001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "TRANS_001"
        assert resp.json()["status"] == "COMPLETED"

    def test_not_found_returns_404(self, client):
        with patch("serving.routers.translation_router.TranslationRepository") as MockRepo:
            MockRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_001/translations/TRANS_999")
        assert resp.status_code == 404


class TestUploadTranslation:
    def test_success(self, client):
        mock_t = _translation()
        with patch("serving.routers.translation_router.TranslationRepository") as MockRepo, \
             patch("serving.routers.translation_router.extract_text_from_upload",
                   new=AsyncMock(return_value="Corrected translation")):
            MockRepo.return_value.update.return_value = mock_t
            resp = client.post(
                "/api/v2/documents/DOC_001/translations/TRANS_001/upload",
                files={"file": ("fix.txt", b"Corrected translation", "text/plain")},
            )
        assert resp.status_code == 200
        assert resp.json()["id"] == "TRANS_001"

    def test_not_found_returns_404(self, client):
        with patch("serving.routers.translation_router.TranslationRepository") as MockRepo, \
             patch("serving.routers.translation_router.extract_text_from_upload",
                   new=AsyncMock(return_value="text")):
            MockRepo.return_value.update.return_value = None
            resp = client.post(
                "/api/v2/documents/DOC_001/translations/TRANS_999/upload",
                files={"file": ("fix.txt", b"text", "text/plain")},
            )
        assert resp.status_code == 404
