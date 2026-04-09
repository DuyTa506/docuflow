"""
OCR Extractor Wrapper.

Wraps the existing process_page_api() output (ServicePageResult / layout_element dicts)
into the unified UnifiedElement format so it can be processed by the same downstream
pipeline as DOCX and PDF-text extractions.

Label map is defined here; the canonical copy is OCR_LABEL_TO_TYPE in core/constants.py.
"""
from typing import List, Dict, Optional, Tuple

from core.models import UnifiedElement

# Map OCR grounding labels → (element_type, heading_level)
# This map also lives in core/constants.OCR_LABEL_TO_TYPE — keep in sync.
_LABEL_MAP: Dict[str, tuple] = {
    "title":     ("heading", 1),
    "sub_title": ("heading", 2),
    "heading":   ("heading", 3),
    "text":      ("text", None),
    "paragraph": ("text", None),
    "table":     ("table", None),
    "figure":    ("figure", None),
    "image":     ("image", None),
    "caption":   ("text", None),
    "list":      ("text", None),
    "footer":    ("text", None),
    "header":    ("text", None),
    "equation":  ("equation", None),
    "formula":   ("equation", None),
}


def ocr_elements_to_unified(
    layout_elements: List[Dict],
    page_number: int,
    source: str = "ocr",
) -> List[UnifiedElement]:
    """
    Convert a list of OCR layout-element dicts (from process_page_api)
    into UnifiedElement instances.

    Args:
        layout_elements: list of dicts with keys: label, text_content/text_full,
                         bbox_x1/y1/x2/y2 or x1/y1/x2/y2, page_number (optional).
        page_number: default page number if not present in individual elements.
        source: source tag, default "ocr".

    Returns:
        List of UnifiedElement in the same order as layout_elements.
    """
    elements: List[UnifiedElement] = []

    for order, elem in enumerate(layout_elements):
        label = elem.get("label", "text")
        elem_type, level = _LABEL_MAP.get(label.lower(), ("text", None))

        # Text content — prefer text_full, fall back to text_content, then text
        text = (
            elem.get("text_full")
            or elem.get("text_content")
            or elem.get("text")
            or ""
        )

        # Bounding box — support both bbox_x1 and x1 key styles
        x1 = elem.get("bbox_x1", elem.get("x1", 0))
        y1 = elem.get("bbox_y1", elem.get("y1", 0))
        x2 = elem.get("bbox_x2", elem.get("x2", 0))
        y2 = elem.get("bbox_y2", elem.get("y2", 0))
        bbox: Optional[Dict] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        pnum = elem.get("page_number", page_number)

        elements.append(UnifiedElement(
            element_type=elem_type,
            text=text.strip(),
            page_number=pnum,
            order=order,
            source=source,
            level=level,
            bbox=bbox,
            font_size=None,
            style_name=None,
        ))

    return elements


class OcrExtractor:
    """
    Thin wrapper that runs the existing DeepSeek vLLM OCR pipeline on a
    single page and returns the result as UnifiedElement instances.

    The raw ServicePageResult (containing image crops, annotated image, etc.)
    is stored as ``self.page_result`` after each call to extract_page() so
    callers can access it without changing the return type.

    Usage:
        extractor = OcrExtractor(client, pdf_path)
        elements = await extractor.extract_page(page_num)
        raw = extractor.page_result  # access raw ServicePageResult
    """

    def __init__(self, client, file_path: str):
        """
        Args:
            client: AsyncOpenAI client configured for the vLLM server.
            file_path: Path to the PDF or image file.
        """
        self.client = client
        self.file_path = file_path
        self.page_result = None  # set after each extract_page() call

    async def extract_page(self, page_number: int) -> List[UnifiedElement]:
        """
        Run OCR on a single page and return UnifiedElements.

        The raw ServicePageResult is stored in ``self.page_result`` for
        callers that need the annotated image or image crops.

        Args:
            page_number: 1-based page number.

        Returns:
            List of UnifiedElement instances (conforming to BaseExtractor ABC).
        """
        from serving.logic import process_page_api

        page_result = None
        async for event in process_page_api(
            client=self.client,
            pdf_path=self.file_path,
            page_num=page_number,
            stream_enabled=False,
        ):
            if event.get("type") == "result":
                page_result = event["result"]

        # Store raw result as attribute — accessible without changing return type
        self.page_result = page_result

        if page_result is None:
            return []

        return ocr_elements_to_unified(
            layout_elements=page_result.layout_elements or [],
            page_number=page_number,
            source="ocr",
        )
