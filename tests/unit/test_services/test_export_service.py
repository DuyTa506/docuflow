"""ExportService cache-key regression tests.

For a non-native (PDF/scan) document, `source=original` must return the raw
uploaded file untouched, while `source=auto`/`extracted` returns a rebuilt
export from OCR text. They must not share a MinIO cache key, or one request
mode can shadow the other's cached result.
"""
from unittest.mock import MagicMock

from services.export_service import ExportService


def _svc():
    svc = ExportService.__new__(ExportService)  # skip __init__ (real MinIO client)
    svc.storage = MagicMock()
    svc.storage.exists.return_value = True  # short-circuit to the cache-hit path
    return svc


def _pdf_doc():
    return MagicMock(id="DOC_1", title="Test Doc", format="pdf", file_path=None)


def test_original_and_auto_source_use_different_cache_keys(monkeypatch):
    svc = _svc()
    doc = _pdf_doc()
    monkeypatch.setattr(svc, "_effective_ocr_mode", lambda db, doc, mode, fmt: "auto")

    key_auto, _, _ = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="docx", source="auto"
    )
    key_original, _, _ = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="docx", source="original"
    )

    assert key_auto != key_original


def test_auto_and_extracted_source_share_cache_key(monkeypatch):
    """Both produce the same rebuilt-from-OCR-text output for a PDF, so
    sharing a cache key is correct (not a regression of the fix above)."""
    svc = _svc()
    doc = _pdf_doc()
    monkeypatch.setattr(svc, "_effective_ocr_mode", lambda db, doc, mode, fmt: "auto")

    key_auto, _, _ = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="docx", source="auto"
    )
    key_extracted, _, _ = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="docx", source="extracted"
    )

    assert key_auto == key_extracted
