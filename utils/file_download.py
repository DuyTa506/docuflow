"""Shared helper for building .docx download responses."""

import io
import unicodedata
from urllib.parse import quote

from docx import Document as DocxDocument
from fastapi import Response


def build_docx_response(
    filename: str,
    content: str,
    *,
    title: str | None = None,
    headings: list[str] | None = None,
) -> Response:
    """Build a .docx Response from text content.

    Args:
        filename: The filename for Content-Disposition header (ASCII-safe).
        content: The text content to include in the document.
        title: Optional level-1 heading at the top of the document.
        headings: Optional additional headings inserted before the content.
    """
    docx = DocxDocument()
    if title:
        docx.add_heading(title, level=1)
    for h in headings or []:
        docx.add_heading(h, level=2)
    for line in content.splitlines():
        docx.add_paragraph(line)

    buf = io.BytesIO()
    docx.save(buf)
    buf.seek(0)

    # Use RFC 5987 filename*=UTF-8''... for non-ASCII filenames
    encoded = quote(filename, safe="")
    disposition = f'attachment; filename="{filename}"; filename*=UTF-8\'\'{encoded}'
    return Response(
        content=buf.read(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": disposition},
    )


def safe_filename(text: str, max_len: int = 60) -> str:
    """Sanitize a string for use in a filename, keeping only ASCII alphanumeric chars."""
    # Normalize unicode (e.g. ị → i) and strip diacritics
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Keep only ASCII letters, digits, space, hyphen, underscore
    return "".join(c if c.isascii() and (c.isalnum() or c in " -_") else "_" for c in text)[:max_len]
