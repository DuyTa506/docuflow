"""MinIO object key conventions."""

from __future__ import annotations


def document_prefix(doc_id: str) -> str:
    return f"documents/{doc_id}/"


def original_key(doc_id: str, filename: str) -> str:
    safe = filename.replace(" ", "_")
    return f"documents/{doc_id}/original/{safe}"


def export_key(doc_id: str, name: str) -> str:
    """e.g. name=normalized_auto.docx or digest.docx"""
    return f"documents/{doc_id}/exports/{name}"


def translation_file_key(doc_id: str, translation_id: str, ext: str) -> str:
    ext = ext.lstrip(".")
    return f"documents/{doc_id}/translations/{translation_id}.{ext}"


def ocr_export_name(
    *,
    content_type: str,
    mode: str,
    fmt: str,
) -> str:
    """Build export object basename for OCR/normalized downloads."""
    return f"{content_type}_{mode}.{fmt}"


def page_image_key(doc_id: str, page_number: int) -> str:
    return f"documents/{doc_id}/pages/{page_number:04d}.jpg"


def layout_crop_key(doc_id: str, page_number: int, seq: int) -> str:
    return f"documents/{doc_id}/crops/{page_number:04d}_{seq:04d}.jpg"


def normalized_content_key(doc_id: str) -> str:
    return f"documents/{doc_id}/content/normalized.md"


def ocr_content_key(doc_id: str) -> str:
    return f"documents/{doc_id}/content/ocr.md"


def summary_export_key(doc_id: str, summary_id: str) -> str:
    return export_key(doc_id, f"summary_{summary_id}.docx")


def tree_data_key(doc_id: str, tree_index_id: str) -> str:
    return f"documents/{doc_id}/tree/{tree_index_id}.json"
