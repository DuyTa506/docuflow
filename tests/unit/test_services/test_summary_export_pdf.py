"""Summary/digest download only ever produced .docx — no format=pdf option
existed, unlike the OCR and translation download endpoints which both
support docx/pdf. Mirrors the translation export's fmt-keyed caching so PDF
and DOCX exports of the same summary don't collide in MinIO.
"""

from unittest.mock import MagicMock, patch

from services.export_service import ExportService


def _svc():
    svc = ExportService.__new__(ExportService)  # skip __init__ (real MinIO client)
    svc.storage = MagicMock()
    return svc


def _doc():
    return MagicMock(id="DOC_1", title="Test Doc")


def test_summary_export_key_differs_by_format():
    assert ExportService.summary_export_key(
        "DOC_1", "SUM_1", "docx"
    ) != ExportService.summary_export_key("DOC_1", "SUM_1", "pdf")


def test_get_or_build_summary_export_pdf_converts_docx(monkeypatch):
    svc = _svc()
    svc.storage.exists.return_value = False
    svc.storage.put_bytes.return_value = "key"

    with patch(
        "services.export_service.docx_bytes_to_pdf_bytes", return_value=b"%PDF-fake"
    ) as mock_convert:
        key, name, media, data = svc.get_or_build_summary_export(
            db=MagicMock(), doc=_doc(), summary_id="SUM_1", content="hello", fmt="pdf"
        )

    mock_convert.assert_called_once()
    assert media == "application/pdf"
    assert name.endswith(".pdf")
    assert data == b"%PDF-fake"
    svc.storage.put_bytes.assert_not_called()


def test_get_or_build_summary_export_docx_skips_conversion():
    svc = _svc()
    svc.storage.exists.return_value = False
    svc.storage.put_bytes.return_value = "key"

    with patch("services.export_service.docx_bytes_to_pdf_bytes") as mock_convert:
        key, name, media, data = svc.get_or_build_summary_export(
            db=MagicMock(), doc=_doc(), summary_id="SUM_1", content="hello", fmt="docx"
        )

    mock_convert.assert_not_called()
    assert name.endswith(".docx")
    assert data is not None
    svc.storage.put_bytes.assert_not_called()
