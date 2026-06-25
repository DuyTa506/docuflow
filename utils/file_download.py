"""Shared helper for building .docx / .pdf download responses."""

import io
import mimetypes
import os
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import quote

from docx import Document as DocxDocument
from fastapi import HTTPException, Response

from config.settings import settings
from utils.markdown_docx import build_docx_bytes_from_markdown, render_layout_elements_to_docx
from utils.markdown_pandoc import markdown_to_docx_bytes
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
    engine: str | None = None,
) -> Response:
    """Build a .docx Response from text content."""
    export_engine = engine or settings.docx_export_engine

    if structured and is_structured_markdown(content) and export_engine != "python":
        body = markdown_to_docx_bytes(
            normalize_ocr_markdown(content),
            title=title,
            headings=headings,
        )
    elif structured and is_structured_markdown(content):
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

    return _docx_bytes_response(filename, body)


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
    return _docx_bytes_response(filename, buf.getvalue())


def build_docx_bytes_from_content(
    content: str,
    *,
    title: str | None = None,
    headings: list[str] | None = None,
    structured: bool = True,
) -> bytes:
    """Return raw .docx bytes (for PDF conversion pipeline)."""
    if structured and is_structured_markdown(content):
        return markdown_to_docx_bytes(
            normalize_ocr_markdown(content),
            title=title,
            headings=headings,
        )
    docx = DocxDocument()
    if title:
        docx.add_heading(title, level=1)
    for h in headings or []:
        docx.add_heading(h, level=2)
    for line in content.splitlines():
        docx.add_paragraph(line)
    buf = io.BytesIO()
    docx.save(buf)
    return buf.getvalue()


def build_docx_bytes_from_elements(elements, *, title: str | None = None) -> bytes:
    from utils.translation_elements import elements_to_views

    docx = DocxDocument()
    if title:
        docx.add_heading(title, level=1)
    render_layout_elements_to_docx(docx, elements_to_views(elements))
    buf = io.BytesIO()
    docx.save(buf)
    return buf.getvalue()


def docx_bytes_to_pdf_bytes(docx_bytes: bytes) -> bytes:
    """Convert DOCX bytes to PDF via LibreOffice headless."""
    from utils.soffice import run_soffice

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src = tmp / "input.docx"
        src.write_bytes(docx_bytes)
        result = run_soffice(
            ["--headless", "--convert-to", "pdf", "--outdir", str(tmp), str(src)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")
        pdf_path = tmp / "input.pdf"
        if not pdf_path.exists():
            raise RuntimeError("LibreOffice produced no PDF output")
        return pdf_path.read_bytes()


def build_pdf_response(filename: str, pdf_bytes: bytes) -> Response:
    if not filename.lower().endswith(".pdf"):
        filename = f"{Path(filename).stem}.pdf"
    encoded = quote(filename, safe="")
    disposition = f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded}'
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": disposition},
    )


def _docx_bytes_response(filename: str, body: bytes) -> Response:
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
