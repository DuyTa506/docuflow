"""Digest ("Tổng thuật") download was docx-only, unlike OCR/translation/
summary which all support format=docx|pdf. Mirrors the summary export fix:
fmt-keyed caching so digest_pdf and digest_docx don't collide in MinIO.
"""

from unittest.mock import MagicMock, patch

from services.export_service import ExportService


def _svc():
    svc = ExportService.__new__(ExportService)  # skip __init__ (real MinIO client)
    svc.storage = MagicMock()
    return svc


def test_digest_export_key_differs_by_format():
    assert ExportService.digest_export_key("DOC_1", "docx") != ExportService.digest_export_key(
        "DOC_1", "pdf"
    )


def test_get_or_build_digest_export_pdf_converts_docx(monkeypatch):
    svc = _svc()
    svc.storage.exists.return_value = False
    svc.storage.put_bytes.return_value = "key"
    monkeypatch.setattr(
        svc,
        "build_digest_export",
        lambda db, document_id, fmt="docx": (
            b"docx-bytes" if fmt == "docx" else b"%PDF-fake",
            f"digest.{fmt}",
            "application/pdf" if fmt == "pdf" else "docx-media",
        ),
    )
    doc = MagicMock(id="DOC_1", title="Test Doc")
    with patch("services.export_service.DocumentRepository") as mock_repo:
        mock_repo.return_value.get.return_value = doc
        key, name, media, data = svc.get_or_build_digest_export(
            db=MagicMock(), document_id="DOC_1", fmt="pdf"
        )

    assert media == "application/pdf"
    assert name.endswith(".pdf")
    assert data == b"%PDF-fake"
    svc.storage.put_bytes.assert_not_called()


def test_get_or_build_digest_export_docx_default(monkeypatch):
    svc = _svc()
    svc.storage.exists.return_value = False
    svc.storage.put_bytes.return_value = "key"
    monkeypatch.setattr(
        svc,
        "build_digest_export",
        lambda db, document_id, fmt="docx": (b"docx-bytes", f"digest.{fmt}", "docx-media"),
    )
    doc = MagicMock(id="DOC_1", title="Test Doc")
    with patch("services.export_service.DocumentRepository") as mock_repo:
        mock_repo.return_value.get.return_value = doc
        key, name, media, data = svc.get_or_build_digest_export(db=MagicMock(), document_id="DOC_1")

    assert name.endswith(".docx")
    assert data == b"docx-bytes"
    svc.storage.put_bytes.assert_not_called()
