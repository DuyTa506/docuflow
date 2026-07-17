"""Export path selection — keep large-document downloads fast."""

from __future__ import annotations

from config.settings import settings
from data.repositories.document_repo import DocumentRepository
from utils.content_storage import read_text_field
from utils.text_assembly import assemble_ocr_from_pages


def resolve_ocr_content(
    repo: DocumentRepository,
    document_id: str,
    dt,
    *,
    content_type: str,
) -> str:
    """Prefer offloaded full-text blob; fall back to per-page assembly."""
    if content_type == "ocr":
        content = read_text_field(
            inline=dt.ocr_content,
            key=dt.ocr_content_key,
        )
        if content:
            return content
        pages = repo.get_pages(document_id)
        return assemble_ocr_from_pages(pages) if pages else ""

    return read_text_field(
        inline=dt.normalized_content,
        key=dt.normalized_content_key,
    )


def spatial_export_plan(
    repo: DocumentRepository,
    document_id: str,
    *,
    mode: str,
    text_overridden: bool,
) -> tuple[bool, bool]:
    if mode in ("plain", "markdown") or text_overridden:
        return False, False

    element_count = repo.count_elements(document_id)
    if element_count <= 0:
        return False, False
    if element_count > settings.ocr_download_spatial_max_elements:
        return False, False

    page_count = repo.count_pages(document_id)
    if page_count > settings.ocr_download_spatial_max_pages:
        return False, False

    embed_images = element_count <= settings.export_spatial_embed_images_max_elements
    return True, embed_images


def translation_routing_allows_spatial(element_count: int) -> bool:
    """Eligibility gate for routing *translation* through layout elements.

    Deliberately independent from ocr_download_spatial_max_elements (a
    download-speed cap): a book just over that cap would silently fall to
    tree/flat translation and lose every figure in the translated artifact,
    even with all crop images stored (DOC_066: 2534 elements, 145 figures).
    """
    return 0 < element_count <= settings.translation_spatial_max_elements


def translation_spatial_plan(
    repo: DocumentRepository,
    document_id: str,
    *,
    element_count: int,
    source: str,
) -> tuple[bool, bool]:
    """Spatial export for translated elements (same caps as OCR spatial)."""
    if source == "flat":
        return False, False
    if element_count <= 0:
        return False, False
    if element_count > settings.ocr_download_spatial_max_elements:
        return False, False
    page_count = repo.count_pages(document_id)
    if page_count > settings.ocr_download_spatial_max_pages:
        return False, False
    embed_images = element_count <= settings.export_spatial_embed_images_max_elements
    return True, embed_images
