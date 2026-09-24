"""Hybrid layout PDF renderer (translation + OCR)."""

from core.pdf_render.geometry import RENDERER_VERSION, PageMeta, PageScene, Rect, Region
from core.pdf_render.quality import PdfRenderQuality
from core.pdf_render.regions import build_page_scene
from core.pdf_render.renderer import RenderResult, render_document_pdf

__all__ = [
    "RENDERER_VERSION",
    "PageMeta",
    "PageScene",
    "Rect",
    "Region",
    "PdfRenderQuality",
    "RenderResult",
    "render_document_pdf",
    "build_page_scene",
]
