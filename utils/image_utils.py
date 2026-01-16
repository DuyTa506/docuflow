"""
Image utilities for OCR workflow.

Handles image loading, conversion, and rendering.
"""
import base64
from io import BytesIO
from typing import Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageOps


def render_pdf_page_to_base64(pdf_path: str, page_num: int, target_dpi: int = 200) -> str:
    """
    Render a PDF page to base64-encoded PNG image.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: 1-indexed page number
        target_dpi: Target DPI for rendering (default 200)
    
    Returns:
        Base64-encoded PNG string
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)  # 0-indexed
    
    # Render at target DPI
    mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert to PIL Image
    img = Image.open(BytesIO(pix.tobytes("png")))
    doc.close()
    
    # Convert to base64
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


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
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    
    # Resize if needed
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert to base64
    buf = BytesIO()
    img.save(buf, format='PNG')
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


def get_image_dimensions(image_or_path) -> Tuple[int, int]:
    """
    Get image dimensions (width, height).
    
    Args:
        image_or_path: PIL Image or path to image file
    
    Returns:
        Tuple of (width, height)
    """
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path)
    else:
        img = image_or_path
    
    return img.size
