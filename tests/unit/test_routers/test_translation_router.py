import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi import HTTPException


@pytest.fixture(autouse=True)
def _authorized_document():
    mock_doc = MagicMock()
    mock_doc.id = "DOC_001"
    mock_doc.user_id = "USR_001"
    mock_doc.title = "Test Doc"
    mock_doc.original_filename = "test.docx"
    with patch(
        "serving.routers.translation_router.get_authorized_document",
        return_value=mock_doc,
    ):
        yield mock_doc


def _translation(tid="TRANS_001"):
    t = MagicMock()
    t.id = tid
    t.document_id = "DOC_001"
    t.target_language = "vi"
    t.translated_content = "Translated text"
    t.translated_file_path = None
    t.translated_elements = None
    t.translation_mode = "flat"
    t.status = "COMPLETED"
    t.created_at = None
    t.updated_at = None
    return t


def _doc():
    d = MagicMock()
    d.id = "DOC_001"
    d.title = "Test Doc"
    d.user_id = "USR_001"
    d.original_filename = "test.docx"
    return d


class TestStartTranslation:
    def test_success(self, client):
        with patch("serving.routers.translation_router._svc") as mock_svc:
            mock_svc.submit.return_value = ("TASK_001", "TRANS_001", False)
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
        with patch(
            "serving.routers.translation_router.get_authorized_document",
            side_effect=HTTPException(status_code=404, detail="Document not found"),
        ):
            resp = client.get("/api/v2/documents/DOC_999/translations")
        assert resp.status_code == 404


class TestGetTranslation:
    def test_success(self, client):
        mock_t = _translation()
        with patch("serving.routers.translation_router.TranslationRepository") as MockRepo, \
             patch("serving.routers.translation_router.DocumentRepository") as MockDocRepo:
            MockRepo.return_value.get.return_value = mock_t
            MockDocRepo.return_value.get_pages.return_value = []
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
                   new=AsyncMock(return_value="Corrected translation")), \
             patch("serving.routers.translation_router.export_service") as mock_exp:
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


class TestDownloadTranslation:
    def test_download_docx_inplace_returns_file(self, client):
        from fastapi.responses import Response

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_t = _translation()
        mock_t.translation_mode = "docx_inplace"
        mock_t.translated_file_path = "documents/DOC_001/translations/TRANS_001.docx"
        with patch("serving.routers.translation_router.asyncio.to_thread", side_effect=_to_thread), \
             patch("serving.routers.translation_router.export_service") as mock_exp, \
             patch(
                 "serving.routers.translation_router.build_stored_file_response",
                 return_value=Response(content=b"PK-translated"),
             ):
            mock_exp.get_or_build_translation_export.return_value = (
                mock_t.translated_file_path,
                "translation_VI_Test Doc.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            with patch("serving.routers.translation_router.TranslationRepository") as MockTransRepo:
                MockTransRepo.return_value.get.return_value = mock_t
                resp = client.get("/api/v2/documents/DOC_001/translations/TRANS_001/download")
        assert resp.status_code == 200
        assert resp.content == b"PK-translated"

    def test_download_element_based_uses_structured_docx(self, client):
        from fastapi.responses import Response

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_t = _translation()
        mock_t.translation_mode = "element_based"
        with patch("serving.routers.translation_router.asyncio.to_thread", side_effect=_to_thread), \
             patch("serving.routers.translation_router.export_service") as mock_exp, \
             patch(
                 "serving.routers.translation_router.build_stored_file_response",
                 return_value=Response(content=b"PK-test"),
             ):
            mock_exp.get_or_build_translation_export.return_value = (
                "documents/DOC_001/translations/TRANS_001.docx",
                "translation_VI_Test Doc.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            with patch("serving.routers.translation_router.TranslationRepository") as MockTransRepo:
                MockTransRepo.return_value.get.return_value = mock_t
                resp = client.get("/api/v2/documents/DOC_001/translations/TRANS_001/download")
        assert resp.status_code == 200

    def test_download_flat_mode(self, client):
        from fastapi.responses import Response

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_t = _translation()
        mock_t.translation_mode = "flat"
        with patch("serving.routers.translation_router.asyncio.to_thread", side_effect=_to_thread), \
             patch("serving.routers.translation_router.export_service") as mock_exp, \
             patch(
                 "serving.routers.translation_router.build_stored_file_response",
                 return_value=Response(content=b"PK-test"),
             ):
            mock_exp.get_or_build_translation_export.return_value = (
                "documents/DOC_001/translations/TRANS_001.docx",
                "translation_VI_Test Doc.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            with patch("serving.routers.translation_router.TranslationRepository") as MockTransRepo:
                MockTransRepo.return_value.get.return_value = mock_t
                resp = client.get(
                    "/api/v2/documents/DOC_001/translations/TRANS_001/download?source=flat"
                )
        assert resp.status_code == 200
