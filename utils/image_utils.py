"""
Image utilities for OCR workflow.

Handles image loading, conversion, and rendering.
"""

import base64
from io import BytesIO
from typing import Any, Optional

import fitz  # PyMuPDF
from PIL import Image, ImageOps


def render_pdf_page_to_jpeg_bytes(
    pdf_path: Optional[str] = None,
    page_num: int = 1,
    *,
    doc: Any = None,
    target_dpi: int = 200,
    max_size: int = 2048,
    quality: int = 95,
) -> bytes:
    """Render a PDF page to JPEG bytes (same pixmap → PIL → JPEG pipeline).

    Pass an already-open ``doc`` to avoid open/close per page inside a worker.
    """
    close = False
    if doc is None:
        if not pdf_path:
            raise ValueError("pdf_path or doc is required")
        doc = fitz.open(pdf_path)
        close = True
    try:
        page = doc.load_page(page_num - 1)  # 0-indexed
        mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    finally:
        if close:
            doc.close()


def render_pdf_page_to_base64(
    pdf_path: str, page_num: int, target_dpi: int = 200, max_size: int = 2048, quality: int = 95
) -> str:
    """
    Render a PDF page to base64-encoded JPEG image.

    Args:
        pdf_path: Path to the PDF file
        page_num: 1-indexed page number
        target_dpi: Target DPI for rendering (default 200)
        max_size: Maximum dimension (width or height) before resizing (default 2048)
        quality: JPEG quality (default 95 -- fine for the OCR model's small,
            capped input images; export call sites with much larger
            dimensions should pass a lower value to avoid multi-MB pages)

    Returns:
        Base64-encoded JPEG string
    """
    return base64.b64encode(
        render_pdf_page_to_jpeg_bytes(
            pdf_path,
            page_num,
            target_dpi=target_dpi,
            max_size=max_size,
            quality=quality,
        )
    ).decode()


def image_to_base64(image_path: str, max_size: int = 2048) -> str:
    """
    Load an image and convert to base64-encoded PNG.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) before resizing

    Returns:
        Base64-encoded PNG string
    """
    img = Image.open(image_path)

    # Fix EXIF orientation
    img = ImageOps.exif_transpose(img)

    # Convert to RGB
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")

    # Resize if needed
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Convert to base64
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def decode_base64_image(b64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.

    Args:
        b64_string: Base64-encoded image string

    Returns:
        PIL Image object
    """
    img_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_data))
