"""Shared helper for building .docx download responses."""

import io
import mimetypes
import os
import unicodedata
from urllib.parse import quote

from docx import Document as DocxDocument
from fastapi import HTTPException, Response

from utils.markdown_docx import build_docx_bytes_from_markdown, render_layout_elements_to_docx
from utils.ocr_markdown import is_structured_markdown, normalize_ocr_markdown

_NATIVE_WORD_FORMATS = frozenset({"docx", "doc"})


def is_native_word_document(doc_format: str | None) -> bool:
    return (doc_format or "").lower() in _NATIVE_WORD_FORMATS


def build_original_file_response(
    file_path: str,
    *,
    download_name: str | None = None,
) -> Response:
    """Stream the uploaded source file as-is (no markdown round-trip)."""
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Original file not found on disk.")

    filename = download_name or os.path.basename(file_path)
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    with open(file_path, "rb") as f:
        body = f.read()

    encoded = quote(filename, safe="")
    disposition = f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded}'
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": disposition},
    )


def build_docx_response(
    filename: str,
    content: str,
    *,
    title: str | None = None,
    headings: list[str] | None = None,
    structured: bool = True,
) -> Response:
    """Build a .docx Response from text content.

    When ``structured`` is True and the content looks like OCR markdown/HTML,
    headings, lists, and tables are rendered into native Word structures.
    Plain text fallbacks to one paragraph per line (legacy behaviour).
    """
    if structured and is_structured_markdown(content):
        body = build_docx_bytes_from_markdown(
            normalize_ocr_markdown(content),
            title=title,
            headings=headings,
        )
    else:
        docx = DocxDocument()
        if title:
            docx.add_heading(title, level=1)
        for h in headings or []:
            docx.add_heading(h, level=2)
        for line in content.splitlines():
            docx.add_paragraph(line)
        buf = io.BytesIO()
        docx.save(buf)
        body = buf.getvalue()

    encoded = quote(filename, safe="")
    disposition = f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded}'
    return Response(
        content=body,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": disposition},
    )


def build_docx_response_from_elements(
    filename: str,
    elements,
    *,
    title: str | None = None,
) -> Response:
    """Build a .docx from spatial layout elements (reading order preserved)."""
    from utils.translation_elements import elements_to_views

    docx = DocxDocument()
    if title:
        docx.add_heading(title, level=1)
    render_layout_elements_to_docx(docx, elements_to_views(elements))

    buf = io.BytesIO()
    docx.save(buf)
    body = buf.getvalue()

    encoded = quote(filename, safe="")
    disposition = f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded}'
    return Response(
        content=body,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": disposition},
    )


def safe_filename(text: str, max_len: int = 60) -> str:
    """Sanitize a string for use in a filename, keeping only ASCII alphanumeric chars."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "".join(c if c.isascii() and (c.isalnum() or c in " -_") else "_" for c in text)[:max_len]
