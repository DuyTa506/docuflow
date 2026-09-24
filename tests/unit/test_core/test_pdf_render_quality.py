"""Quality aggregation and cache-key versioning."""

from core.pdf_render.geometry import RENDERER_VERSION
from core.pdf_render.quality import QualityIssue, aggregate_quality
from utils.export_paths import (
    overlay_rollback_enabled,
    resolve_ocr_pdf_mode,
    resolve_translation_pdf_mode,
)
from utils.storage_keys import ocr_export_name, translation_file_key


def test_critical_overlap_triggers_reflow_fallback():
    issues = [QualityIssue(1, "column_overlap", "boom", critical=True)]
    report = aggregate_quality(issues, 1, "layout")
    assert report.ok is False
    assert report.fallback == "reflow"


def test_ocr_auto_prefers_facsimile():
    assert resolve_ocr_pdf_mode("auto", has_spatial=True) == "facsimile"
    assert resolve_ocr_pdf_mode("auto", has_spatial=False) == "reflow"


def test_translation_auto_prefers_layout():
    assert resolve_translation_pdf_mode("auto", has_spatial=True) == "layout"
    assert resolve_translation_pdf_mode("reflow", has_spatial=True) == "reflow"


def test_overlay_rollback_off_by_default():
    assert overlay_rollback_enabled() is False


def test_versioned_translation_pdf_key():
    key = translation_file_key("DOC_1", "T1", "pdf", pdf_mode="layout")
    assert RENDERER_VERSION in key
    assert key.endswith(".layout.pdf")
    legacy = translation_file_key("DOC_1", "T1", "pdf")
    assert legacy.endswith("T1.pdf")


def test_ocr_export_name_includes_mode_and_version():
    name = ocr_export_name(content_type="ocr", mode="layout", fmt="pdf", pdf_mode="facsimile")
    assert "facsimile" in name
    assert RENDERER_VERSION in name
