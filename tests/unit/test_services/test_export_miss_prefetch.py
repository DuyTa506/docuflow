"""Export cache miss returns RAM bytes without requiring MinIO put first;
prefetch warms UI PDF modes (OCR facsimile / translation layout)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.export_service import ExportService


def _svc():
    svc = ExportService.__new__(ExportService)
    svc.storage = MagicMock()
    return svc


@pytest.mark.asyncio
async def test_ocr_miss_returns_bytes_without_put(monkeypatch):
    svc = _svc()
    svc.storage.exists.return_value = False
    monkeypatch.setattr(
        svc,
        "_effective_ocr_mode",
        lambda db, doc, mode, fmt="docx": "markdown",
    )
    monkeypatch.setattr(
        svc,
        "build_ocr_export",
        lambda *a, **k: (b"docx-bytes", "ocr_Doc.docx", "application/octet-stream"),
    )
    doc = MagicMock(id="DOC_1", title="Doc", format="pdf", file_path=None)

    key, name, media, data = svc.get_or_build_ocr_export(
        db=MagicMock(),
        doc=doc,
        content_type="ocr",
        mode="auto",
        fmt="docx",
        source="extracted",
    )

    assert data == b"docx-bytes"
    assert name.endswith(".docx")
    assert key
    svc.storage.put_bytes.assert_not_called()
    # Caller (router / prefetch) persists — simulate put_export
    svc.put_export(key, data, content_type=media)
    svc.storage.put_bytes.assert_called_once()


@pytest.mark.asyncio
async def test_cache_ocr_prefetch_uses_facsimile_pdf_mode():
    svc = _svc()
    calls = []

    def _capture(*args, **kwargs):
        calls.append(kwargs)
        return ("k", "n", "m", b"data")

    svc.get_or_build_ocr_export = MagicMock(side_effect=_capture)
    svc.put_export = MagicMock()

    doc = MagicMock(id="DOC_1", format="pdf")
    with (
        patch("services.export_service.DocumentRepository") as mock_repo,
        patch("services.export_service.get_db_manager") as mock_dbm,
        patch("services.export_service.is_native_word_document", return_value=False),
        patch("services.export_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
    ):
        mock_repo.return_value.get.return_value = doc
        session = MagicMock()
        mock_dbm.return_value.session.return_value.__enter__.return_value = session
        mock_dbm.return_value.session.return_value.__exit__.return_value = None

        async def _thread(fn, *a, **k):
            return fn(*a, **k)

        mock_thread.side_effect = _thread
        await svc.cache_ocr_exports_after_extract(MagicMock(), "DOC_1")

    pdf_calls = [c for c in calls if c.get("fmt") == "pdf"]
    assert pdf_calls
    assert all(c.get("pdf_mode") == "facsimile" for c in pdf_calls)
    assert svc.put_export.called


@pytest.mark.asyncio
async def test_cache_translation_prefetch_uses_layout_pdf_mode():
    svc = _svc()
    calls = []

    def _capture(*args, **kwargs):
        calls.append(kwargs)
        return ("k", "n", "m", b"data")

    svc.get_or_build_translation_export = MagicMock(side_effect=_capture)
    svc.put_export = MagicMock()

    with (
        patch("services.export_service.DocumentRepository") as mock_doc_repo,
        patch("services.export_service.TranslationRepository") as mock_trans_repo,
        patch("services.export_service.get_db_manager") as mock_dbm,
        patch("services.export_service.asyncio.to_thread", new_callable=AsyncMock) as mock_thread,
    ):
        mock_doc_repo.return_value.get.return_value = MagicMock(id="DOC_1")
        mock_trans_repo.return_value.get.return_value = MagicMock(id="TR_1")
        mock_dbm.return_value.session.return_value.__enter__.return_value = MagicMock()
        mock_dbm.return_value.session.return_value.__exit__.return_value = None

        async def _thread(fn, *a, **k):
            return fn(*a, **k)

        mock_thread.side_effect = _thread
        await svc.cache_translation_exports(MagicMock(), "DOC_1", "TR_1")

    pdf_calls = [c for c in calls if c.get("fmt") == "pdf"]
    assert pdf_calls
    assert all(c.get("pdf_mode") == "layout" for c in pdf_calls)
    assert svc.put_export.called
