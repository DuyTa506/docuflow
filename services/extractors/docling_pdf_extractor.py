"""
PDF text + image extraction via docling-parse.
"""

import base64
import io
from typing import Dict, List

from core.models import UnifiedElement


class DoclingPdfExtractor:
    """Extract textlines and embedded images from a PDF via docling-parse."""

    def __init__(self, file_path: str):
        from docling_parse.pdf_parser import DoclingPdfParser

        self.file_path = file_path
        self._parser = DoclingPdfParser()
        self._doc = self._parser.load(file_path)

    @property
    def total_pages(self) -> int:
        return self._doc.number_of_pages()

    def extract_page(self, page_number: int) -> List[UnifiedElement]:
        """Extract all textlines and images from a single page."""
        page = self._doc.get_page(page_number)
        page_h = page.dimension.height
        elements: List[UnifiedElement] = []

        for cell in page.textline_cells:
            text = cell.text or ""
            if not text.strip():
                continue
            elements.append(
                UnifiedElement(
                    element_type="text",
                    text=text,
                    page_number=page_number,
                    order=len(elements),
                    source="docling",
                    bbox=self._to_bbox_dict(cell.rect, page_h),
                )
            )

        for idx, img in enumerate(page.bitmap_resources):
            b64 = None
            width_px = None
            height_px = None

            if img.image is not None:
                try:
                    pil_img = img.image.pil_image
                    if pil_img is not None:
                        width_px, height_px = pil_img.size
                        buf = io.BytesIO()
                        pil_img.convert("RGB").save(buf, format="JPEG", quality=85)
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                except Exception:
                    pass

            elements.append(
                UnifiedElement(
                    element_type="image",
                    text=f"(img_content)[image_{idx + 1}]",
                    page_number=page_number,
                    order=len(elements),
                    source="docling",
                    bbox=self._to_bbox_dict(img.rect, page_h),
                    image_bytes_b64=b64,
                    image_width=width_px,
                    image_height=height_px,
                )
            )

        return elements

    @staticmethod
    def _to_bbox_dict(rect, page_height: float) -> dict:
        """Convert BoundingRectangle (bottom-left origin) → {x1, y1, x2, y2} top-left."""
        tl = rect.to_top_left_origin(page_height=page_height)
        bb = tl.to_bounding_box()
        return {"x1": bb.l, "y1": bb.t, "x2": bb.r, "y2": bb.b}


def classify_pages(docling_doc, threshold: int = 50) -> Dict[int, str]:
    """Classify each page as 'text' or 'scanned' based on extracted text length."""
    result: Dict[int, str] = {}
    for page_no, page in docling_doc.iterate_pages():
        total_chars = sum(len(c.text or "") for c in page.textline_cells)
        result[page_no] = "text" if total_chars >= threshold else "scanned"
    return result
