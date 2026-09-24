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

    key_auto, _, _, _ = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="docx", source="auto"
    )
    key_original, _, _, _ = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="docx", source="original"
    )

    assert key_auto != key_original


def test_auto_and_extracted_source_share_cache_key(monkeypatch):
    """Both produce the same rebuilt-from-OCR-text output for a PDF, so
    sharing a cache key is correct (not a regression of the fix above)."""
    svc = _svc()
    doc = _pdf_doc()
    monkeypatch.setattr(svc, "_effective_ocr_mode", lambda db, doc, mode, fmt: "auto")

    key_auto, _, _, _ = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="docx", source="auto"
    )
    key_extracted, _, _, _ = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="docx", source="extracted"
    )

    assert key_auto == key_extracted


def test_a_reflow_fallback_is_not_stored_under_the_facsimile_key(monkeypatch):
    """`auto` may hand back a reflow text PDF; the facsimile slot must not hold it.

    With the key resolved from the request ("auto" → "facsimile"), one auto
    request whose quality check failed stored a text-only PDF under the
    facsimile key, and every later facsimile download served it. Seen on
    DOC_028: a 5-page scan came back as 42 KB of text, no page images.
    """
    import utils.export_paths as export_paths

    svc = _svc()
    svc.storage.exists.return_value = False  # nothing cached: take the build path
    doc = _pdf_doc()
    monkeypatch.setattr(svc, "_effective_ocr_mode", lambda db, doc, mode, fmt: "auto")
    monkeypatch.setattr("services.export_service.DocumentRepository", MagicMock())
    monkeypatch.setattr(export_paths, "spatial_export_plan", lambda *a, **kw: (True, None))
    monkeypatch.setattr(
        svc, "build_ocr_export", lambda *a, **kw: (b"%PDF", "ocr_x.reflow.pdf", "application/pdf")
    )

    key, _, _, data = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="pdf", source="auto"
    )

    assert data == b"%PDF"
    assert key.endswith("_reflow_v3.pdf") or "reflow" in key
    assert "facsimile" not in key


def test_a_facsimile_render_keeps_the_facsimile_key(monkeypatch):
    import utils.export_paths as export_paths

    svc = _svc()
    svc.storage.exists.return_value = False
    doc = _pdf_doc()
    monkeypatch.setattr(svc, "_effective_ocr_mode", lambda db, doc, mode, fmt: "auto")
    monkeypatch.setattr("services.export_service.DocumentRepository", MagicMock())
    monkeypatch.setattr(export_paths, "spatial_export_plan", lambda *a, **kw: (True, None))
    monkeypatch.setattr(
        svc,
        "build_ocr_export",
        lambda *a, **kw: (b"%PDF", "ocr_x.facsimile.pdf", "application/pdf"),
    )

    key, _, _, _ = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="pdf", source="auto"
    )

    assert "facsimile" in key


def test_an_auto_request_reuses_a_cached_reflow_fallback(monkeypatch):
    """Otherwise every auto download rebuilds the facsimile just to discard it."""
    import utils.export_paths as export_paths

    svc = _svc()
    doc = _pdf_doc()
    monkeypatch.setattr(svc, "_effective_ocr_mode", lambda db, doc, mode, fmt: "auto")
    monkeypatch.setattr("services.export_service.DocumentRepository", MagicMock())
    monkeypatch.setattr(export_paths, "spatial_export_plan", lambda *a, **kw: (True, None))
    svc.storage.exists.side_effect = lambda key: "reflow" in key

    key, _, _, data = svc.get_or_build_ocr_export(
        db=MagicMock(), doc=doc, content_type="ocr", mode="auto", fmt="pdf", source="auto"
    )

    assert "reflow" in key
    assert data is None  # served from the cache
