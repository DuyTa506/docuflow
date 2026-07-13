from unittest.mock import AsyncMock, MagicMock, mock_open, patch

from fastapi import HTTPException


def _doc(doc_id="DOC_001", user_id="USR_001"):
    d = MagicMock()
    d.id = doc_id
    d.title = "Test Doc"
    d.original_filename = "test.pdf"
    d.format = "pdf"
    d.file_type = "pdf"
    d.total_pages = 5
    d.processing_status = "EXTRACTED"
    d.source_language = "en"
    d.user_id = user_id
    d.file_path = None
    d.created_at = None
    d.updated_at = None
    return d


def _dt():
    dt = MagicMock()
    dt.ocr_content = "Raw OCR text"
    dt.normalized_content = "Normalized text"
    dt.ocr_content_key = None
    dt.normalized_content_key = None
    return dt


class TestUploadDocument:
    def test_success_returns_201(self, client):
        mock_doc = _doc()
        with (
            patch("serving.routers.documents_router._doc_svc") as mock_svc,
            patch("serving.routers.documents_router.os.makedirs"),
            patch("serving.routers.documents_router.shutil.copyfileobj"),
            patch("builtins.open", mock_open()),
        ):
            mock_svc.upload_document.return_value = mock_doc
            resp = client.post(
                "/api/v2/documents/upload",
                files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
                data={"title": "Test Doc"},
            )
        assert resp.status_code == 201
        assert resp.json()["document_id"] == "DOC_001"

    def test_unsupported_type_returns_400(self, client):
        with patch("serving.routers.documents_router.os.makedirs"):
            resp = client.post(
                "/api/v2/documents/upload",
                files={"file": ("virus.exe", b"binary", "application/octet-stream")},
            )
        assert resp.status_code == 400

    def test_upload_auto_triggers_extraction(self, client):
        """Uploading a document must kick off OCR/extraction immediately —
        the user shouldn't need a second click (or the FE a second call)."""
        mock_doc = _doc()
        with (
            patch("serving.routers.documents_router._doc_svc") as mock_svc,
            patch("serving.routers.documents_router.os.makedirs"),
            patch("serving.routers.documents_router.shutil.copyfileobj"),
            patch("builtins.open", mock_open()),
        ):
            mock_svc.upload_document.return_value = mock_doc
            mock_svc.submit_extraction_async = AsyncMock(return_value=("EXTRACT_007", False))
            resp = client.post(
                "/api/v2/documents/upload",
                files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
            )
        assert resp.status_code == 201
        mock_svc.submit_extraction_async.assert_awaited_once()
        assert mock_svc.submit_extraction_async.await_args.args[1] == "DOC_001"
        assert resp.json()["extract_task_id"] == "EXTRACT_007"

    def test_upload_survives_extraction_submit_failure(self, client):
        """A Temporal outage must not fail the upload itself — the document
        is saved and OCR can be started manually later."""
        mock_doc = _doc()
        with (
            patch("serving.routers.documents_router._doc_svc") as mock_svc,
            patch("serving.routers.documents_router.os.makedirs"),
            patch("serving.routers.documents_router.shutil.copyfileobj"),
            patch("builtins.open", mock_open()),
        ):
            mock_svc.upload_document.return_value = mock_doc
            mock_svc.submit_extraction_async = AsyncMock(side_effect=RuntimeError("temporal down"))
            resp = client.post(
                "/api/v2/documents/upload",
                files={"file": ("test.pdf", b"%PDF-1.4", "application/pdf")},
            )
        assert resp.status_code == 201
        assert resp.json()["extract_task_id"] is None


class TestStartExtraction:
    def test_success(self, client):
        with patch("serving.routers.documents_router._doc_svc") as mock_svc:
            mock_svc.submit_extraction_async = AsyncMock(return_value=("TASK_001", False))
            resp = client.post("/api/v2/documents/DOC_001/extract")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == "TASK_001"

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.documents_router._doc_svc") as mock_svc:
            mock_svc.submit_extraction_async = AsyncMock(side_effect=ValueError("Not found"))
            resp = client.post("/api/v2/documents/DOC_999/extract")
        assert resp.status_code == 404


class TestListDocuments:
    def test_member_sees_own_docs(self, client):
        mock_doc = _doc()
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_for_user.return_value = [mock_doc]
            mock_repo.count_for_user.return_value = 1
            resp = client.get("/api/v2/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["page"] == 1
        assert data["total_pages"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == "DOC_001"

    def test_admin_sees_all_docs(self, admin_client):
        mock_doc = _doc()
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.list.return_value = [mock_doc]
            mock_repo.count.return_value = 1
            resp = admin_client.get("/api/v2/documents")
        assert resp.status_code == 200
        mock_repo.list.assert_called_once()
        assert resp.json()["total"] == 1

    def test_pagination_total_pages_calculation(self, client):
        docs = [_doc(f"DOC_{i:03d}") for i in range(5)]
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_for_user.return_value = docs
            mock_repo.count_for_user.return_value = 23
            resp = client.get("/api/v2/documents?page=2&limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 23
        assert data["page"] == 2
        assert data["total_pages"] == 5  # ceil(23/5)
        assert data["limit"] == 5

    def test_page_param_accepted(self, client):
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_for_user.return_value = []
            mock_repo.count_for_user.return_value = 0
            resp = client.get("/api/v2/documents?page=3&limit=10")
        assert resp.status_code == 200


class TestGetDocument:
    def test_success(self, client):
        mock_doc = _doc()
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            MockRepo.return_value.get.return_value = mock_doc
            resp = client.get("/api/v2/documents/DOC_001")
        assert resp.status_code == 200
        assert resp.json()["id"] == "DOC_001"

    def test_not_found_returns_404(self, client):
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            MockRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_999")
        assert resp.status_code == 404

    def test_other_users_doc_returns_403(self, client):
        with patch(
            "serving.routers.documents_router.get_authorized_document",
            side_effect=HTTPException(status_code=403, detail="Access denied"),
        ):
            resp = client.get("/api/v2/documents/DOC_001")
        assert resp.status_code == 403


class TestGetDocumentText:
    def test_success(self, client):
        mock_doc = _doc()
        mock_dt = _dt()
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.get.return_value = mock_doc
            mock_repo.get_digitized_text.return_value = mock_dt
            mock_repo.get_pages.return_value = []
            resp = client.get("/api/v2/documents/DOC_001/text")
        assert resp.status_code == 200
        assert resp.json()["ocr_content"] == "Raw OCR text"
        assert resp.json()["normalized_content"] == "Normalized text"

    def test_document_not_found_returns_404(self, client):
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            MockRepo.return_value.get.return_value = None
            resp = client.get("/api/v2/documents/DOC_999/text")
        assert resp.status_code == 404

    def test_access_denied_returns_403(self, client):
        with patch(
            "serving.routers.documents_router.get_authorized_document",
            side_effect=HTTPException(status_code=403, detail="Access denied"),
        ):
            resp = client.get("/api/v2/documents/DOC_001/text")
        assert resp.status_code == 403

    def test_preview_pages_fetches_only_requested_pages(self, client):
        """preview_pages must limit the DB fetch, not just truncate after
        loading every page (the bug behind the 'slow analysis load' report)."""
        mock_doc = _doc()
        mock_dt = _dt()
        preview_rows = [
            MagicMock(page_number=1, markdown_content="Page one"),
            MagicMock(page_number=2, markdown_content="Page two"),
        ]
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.get.return_value = mock_doc
            mock_repo.get_digitized_text.return_value = mock_dt
            mock_repo.count_pages.return_value = 800
            mock_repo.get_pages.return_value = preview_rows
            resp = client.get("/api/v2/documents/DOC_001/text?preview_pages=2")

        assert resp.status_code == 200
        mock_repo.get_pages.assert_called_once_with("DOC_001", limit=2)
        data = resp.json()
        assert data["total_pages"] == 800
        assert data["truncated"] is True

    def test_no_preview_pages_fetches_all_pages(self, client):
        mock_doc = _doc()
        mock_dt = _dt()
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.get.return_value = mock_doc
            mock_repo.get_digitized_text.return_value = mock_dt
            mock_repo.get_pages.return_value = []
            resp = client.get("/api/v2/documents/DOC_001/text")

        assert resp.status_code == 200
        mock_repo.get_pages.assert_called_once_with("DOC_001")
        mock_repo.count_pages.assert_not_called()


class TestDownloadDocumentText:
    def test_download_docx_returns_original_file(self, client):
        from fastapi.responses import Response

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_doc = _doc()
        mock_doc.format = "docx"
        mock_doc.original_filename = "report.docx"
        mock_doc.file_path = "documents/DOC_001/original/report.docx"
        with (
            patch("serving.routers.documents_router.asyncio.to_thread", side_effect=_to_thread),
            patch("serving.routers.documents_router.export_service") as mock_exp,
            patch(
                "serving.routers.documents_router.build_stored_file_response",
                return_value=Response(content=b"PK-original-docx"),
            ),
        ):
            mock_exp.get_or_build_ocr_export.return_value = (
                mock_doc.file_path,
                "report.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            with patch(
                "serving.routers.documents_router.get_authorized_document",
                return_value=mock_doc,
            ):
                resp = client.get("/api/v2/documents/DOC_001/text/download")
        assert resp.status_code == 200
        assert resp.content == b"PK-original-docx"

    def test_download_docx_can_force_extracted_export(self, client):
        from fastapi.responses import Response

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_doc = _doc()
        mock_doc.format = "docx"
        with (
            patch("serving.routers.documents_router.asyncio.to_thread", side_effect=_to_thread),
            patch("serving.routers.documents_router.export_service") as mock_exp,
            patch(
                "serving.routers.documents_router.build_stored_file_response",
                return_value=Response(content=b"PK-test"),
            ),
        ):
            mock_exp.get_or_build_ocr_export.return_value = (
                "documents/DOC_001/exports/ocr_markdown.docx",
                "ocr_Test Doc.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
                MockRepo.return_value.get.return_value = mock_doc
                resp = client.get(
                    "/api/v2/documents/DOC_001/text/download?source=extracted&type=ocr"
                )
        assert resp.status_code == 200

    def test_download_ocr_returns_docx(self, client):
        from fastapi.responses import Response

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_doc = _doc()
        with (
            patch("serving.routers.documents_router.asyncio.to_thread", side_effect=_to_thread),
            patch("serving.routers.documents_router.export_service") as mock_exp,
            patch(
                "serving.routers.documents_router.build_stored_file_response",
                return_value=Response(content=b"PK-test"),
            ),
        ):
            mock_exp.get_or_build_ocr_export.return_value = (
                "documents/DOC_001/exports/ocr_markdown.docx",
                "ocr_Test Doc.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
                MockRepo.return_value.get.return_value = mock_doc
                resp = client.get("/api/v2/documents/DOC_001/text/download?type=ocr")
        assert resp.status_code == 200

    def test_download_normalized_returns_docx(self, client):
        from fastapi.responses import Response

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_doc = _doc()
        with (
            patch("serving.routers.documents_router.asyncio.to_thread", side_effect=_to_thread),
            patch("serving.routers.documents_router.export_service") as mock_exp,
            patch(
                "serving.routers.documents_router.build_stored_file_response",
                return_value=Response(content=b"PK-test"),
            ),
        ):
            mock_exp.get_or_build_ocr_export.return_value = (
                "documents/DOC_001/exports/normalized_markdown.docx",
                "normalized_Test Doc.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
                MockRepo.return_value.get.return_value = mock_doc
                resp = client.get("/api/v2/documents/DOC_001/text/download?type=normalized")
        assert resp.status_code == 200

    def test_no_extracted_text_returns_404(self, client):
        mock_doc = _doc()

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        with (
            patch("serving.routers.documents_router.asyncio.to_thread", side_effect=_to_thread),
            patch("serving.routers.documents_router.export_service") as mock_exp,
        ):
            mock_exp.get_or_build_ocr_export.side_effect = ValueError("No extracted text found")
            with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
                MockRepo.return_value.get.return_value = mock_doc
                resp = client.get("/api/v2/documents/DOC_001/text/download")
        assert resp.status_code == 404

    def test_access_denied_returns_403(self, client):
        with patch(
            "serving.routers.documents_router.get_authorized_document",
            side_effect=HTTPException(status_code=403, detail="Access denied"),
        ):
            resp = client.get("/api/v2/documents/DOC_001/text/download")
        assert resp.status_code == 403

    def test_build_stored_file_response_is_threaded(self, client):
        """build_stored_file_response does a blocking MinIO existence check —
        it must run via asyncio.to_thread, not directly on the event loop."""
        from fastapi.responses import Response

        async def _to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        mock_doc = _doc()
        with (
            patch(
                "serving.routers.documents_router.asyncio.to_thread", side_effect=_to_thread
            ) as mock_to_thread,
            patch("serving.routers.documents_router.export_service") as mock_exp,
            patch(
                "serving.routers.documents_router.build_stored_file_response",
                return_value=Response(content=b"data"),
            ) as mock_build_resp,
            patch(
                "serving.routers.documents_router.get_authorized_document",
                return_value=mock_doc,
            ),
        ):
            mock_exp.get_or_build_ocr_export.return_value = (
                "documents/DOC_001/exports/ocr_auto.docx",
                "ocr_Test Doc.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            resp = client.get("/api/v2/documents/DOC_001/text/download")

        assert resp.status_code == 200
        threaded_targets = [c.args[0] for c in mock_to_thread.call_args_list]
        assert mock_build_resp in threaded_targets


class TestUploadDocumentText:
    def test_success(self, client):
        mock_doc = _doc()
        mock_dt = _dt()
        with (
            patch("serving.routers.documents_router.DocumentRepository") as MockRepo,
            patch(
                "serving.routers.documents_router.extract_text_from_upload",
                new=AsyncMock(return_value="Corrected text"),
            ),
            patch("serving.routers.documents_router.export_service") as mock_exp,
        ):
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.get.return_value = mock_doc
            mock_repo.get_digitized_text.return_value = mock_dt
            mock_repo.update_digitized_text.return_value = mock_dt
            resp = client.post(
                "/api/v2/documents/DOC_001/text/upload",
                files={"file": ("fix.txt", b"Corrected text", "text/plain")},
            )
        assert resp.status_code == 200
        assert resp.json()["document_id"] == "DOC_001"

    def test_no_digitized_text_returns_409(self, client):
        mock_doc = _doc()
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.get.return_value = mock_doc
            mock_repo.get_digitized_text.return_value = None
            resp = client.post(
                "/api/v2/documents/DOC_001/text/upload",
                files={"file": ("fix.txt", b"text", "text/plain")},
            )
        assert resp.status_code == 409


class TestGetDocumentPages:
    def test_success(self, client):
        mock_doc = _doc()
        mock_page = MagicMock()
        mock_page.id = "PAGE_001"
        mock_page.page_number = 1
        mock_page.markdown_content = "# Heading"
        mock_page.image_width = 800
        mock_page.image_height = 1200
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.get.return_value = mock_doc
            mock_repo.get_pages.return_value = [mock_page]
            resp = client.get("/api/v2/documents/DOC_001/pages")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["page_number"] == 1

    def test_access_denied_returns_403(self, client):
        with patch(
            "serving.routers.documents_router.get_authorized_document",
            side_effect=HTTPException(status_code=403, detail="Access denied"),
        ):
            resp = client.get("/api/v2/documents/DOC_001/pages")
        assert resp.status_code == 403


class TestGetDocumentElements:
    def test_success(self, client, mock_db):
        mock_doc = _doc()
        mock_elem = MagicMock()
        mock_elem.id = "ELEM_001"
        mock_elem.label = "title"
        mock_elem.text_content = "Title text"
        mock_elem.bbox_x1 = 10.0
        mock_elem.bbox_y1 = 20.0
        mock_elem.bbox_x2 = 100.0
        mock_elem.bbox_y2 = 50.0
        mock_elem.bbox_norm_x1 = None
        mock_elem.page_id = "PAGE_001"
        mock_elem.sequence_order = 1
        mock_elem.crop_image_base64 = None

        mock_page = MagicMock()
        mock_page.page_number = 1
        mock_db.query.return_value.filter.return_value.first.return_value = mock_page

        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.get.return_value = mock_doc
            mock_repo.get_elements.return_value = [mock_elem]
            resp = client.get("/api/v2/documents/DOC_001/elements")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["label"] == "title"

    def test_filter_by_label(self, client, mock_db):
        mock_doc = _doc()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        with patch("serving.routers.documents_router.DocumentRepository") as MockRepo:
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_repo.get.return_value = mock_doc
            mock_repo.get_elements.return_value = []
            resp = client.get("/api/v2/documents/DOC_001/elements?label=title")
        assert resp.status_code == 200
        mock_repo.get_elements.assert_called_once_with("DOC_001", label="title")


class TestDeleteDocument:
    def test_success_returns_204(self, client, mock_db):
        mock_doc = _doc()
        with (
            patch("serving.routers.documents_router.DocumentRepository") as MockRepo,
            patch("serving.routers.documents_router.delete_document_cascade") as mock_delete,
            patch("serving.routers.documents_router.export_service") as mock_exp,
            patch(
                "serving.routers.documents_router.get_authorized_document",
                return_value=mock_doc,
            ),
        ):
            mock_repo = MagicMock()
            MockRepo.return_value = mock_repo
            mock_delete.return_value = True
            resp = client.delete("/api/v2/documents/DOC_001")
        assert resp.status_code == 204
        mock_delete.assert_called_once_with("DOC_001")

    def test_not_found_returns_404(self, client):
        with patch(
            "serving.routers.documents_router.get_authorized_document",
            side_effect=HTTPException(status_code=404, detail="Document not found"),
        ):
            resp = client.delete("/api/v2/documents/DOC_999")
        assert resp.status_code == 404

    def test_access_denied_returns_403(self, client):
        with patch(
            "serving.routers.documents_router.get_authorized_document",
            side_effect=HTTPException(status_code=403, detail="Access denied"),
        ):
            resp = client.delete("/api/v2/documents/DOC_001")
        assert resp.status_code == 403
