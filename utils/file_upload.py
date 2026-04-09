"""
Utility for extracting plain text from user-uploaded correction files.

Supported formats:
  .txt  — read as UTF-8 text
  .docx — extract paragraph text via python-docx

Used by translation, summary, and OCR-text upload/override endpoints.
"""
from fastapi import UploadFile, HTTPException


ALLOWED_EXTENSIONS = {".txt", ".docx"}


async def extract_text_from_upload(file: UploadFile) -> str:
    """
    Read an uploaded .txt or .docx file and return its plain text content.

    Args:
        file: FastAPI UploadFile from a multipart/form-data request.

    Returns:
        Extracted text string (stripped of leading/trailing whitespace).

    Raises:
        HTTPException 400: if the file type is unsupported or content is empty.
    """
    filename = file.filename or ""
    ext = _get_extension(filename)

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: .txt, .docx",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if ext == ".txt":
        text = _read_txt(raw_bytes)
    else:
        text = _read_docx(raw_bytes)

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text content found in uploaded file.")

    return text


# ── Private helpers ──────────────────────────────────────────────────

def _get_extension(filename: str) -> str:
    import os
    return os.path.splitext(filename.lower())[1]


def _read_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _read_docx(data: bytes) -> str:
    try:
        import docx
        from io import BytesIO
        doc = docx.Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="python-docx is not installed. Cannot read .docx files.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse .docx file: {exc}",
        )
