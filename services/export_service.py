"""Build and cache document exports in MinIO."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from config.settings import settings
from data.database import get_db_manager
from data.db_models import Document, Translation
from data.repositories import DocumentRepository, TranslationRepository
from services.digest_renderer import DigestRenderer
from services.digest_service import DigestService
from services.object_storage import get_object_storage
from utils.file_download import (
    build_docx_bytes_from_content,
    build_docx_bytes_from_elements,
    build_pdf_bytes_from_elements,
    docx_bytes_to_pdf_bytes,
    extract_pdf_text,
    is_native_word_document,
    safe_filename,
)
from utils.storage_keys import (
    document_prefix,
    export_key,
    ocr_export_name,
    summary_export_key,
    translation_file_key,
)
from utils.translation_elements import deserialize_translated_elements

logger = logging.getLogger(__name__)

_digest_renderer = DigestRenderer()
_digest_service = DigestService()


def _resolve_export_page_backgrounds(doc: Document, pages) -> dict:
    """Best-effort higher-DPI page backgrounds for layout PDF export, re-rendered
    from the original PDF instead of reusing the OCR model's low-res input image.
    Returns {} (never raises) when the original isn't a resolvable PDF.

    Results are cached in MinIO under ``documents/{id}/export_bg/...`` so
    repeated OCR/translation PDF builds reuse the same JPEG bytes. Fully cached
    hits skip downloading the original PDF.
    """
    if doc.format != "pdf" or not doc.file_path:
        return {}
    from utils.layout_pdf import get_or_render_export_backgrounds

    page_numbers = [
        (p.get("page_number") if isinstance(p, dict) else getattr(p, "page_number", None))
        for p in pages
    ]
    page_numbers = [pn for pn in page_numbers if pn]
    if not page_numbers:
        return {}

    # Cache-only pass first — avoid resolve_local_or_key when every page is warm.
    try:
        cached = get_or_render_export_backgrounds(doc.id, None, page_numbers)
        if len(cached) == len(page_numbers):
            return cached
    except Exception:
        logger.debug("export_bg cache-only probe failed for %s", doc.id, exc_info=True)

    storage = get_object_storage()
    local_path = None
    cleanup = False
    try:
        local_path = storage.resolve_local_or_key(doc.file_path)
        cleanup = local_path != doc.file_path
        return get_or_render_export_backgrounds(doc.id, local_path, page_numbers)
    except Exception:
        logger.debug("export background resolution failed for %s", doc.id, exc_info=True)
        return {}
    finally:
        if cleanup and local_path and os.path.isfile(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass


class ExportService:
    """Generate DOCX/PDF exports and persist them in MinIO."""

    def __init__(self) -> None:
        self.storage = get_object_storage()

    # ── Key helpers ───────────────────────────────────────────────────

    @staticmethod
    def ocr_export_key(
        document_id: str, *, content_type: str, mode: str, fmt: str, pdf_mode: str | None = None
    ) -> str:
        name = ocr_export_name(content_type=content_type, mode=mode, fmt=fmt, pdf_mode=pdf_mode)
        return export_key(document_id, name)

    @staticmethod
    def translation_export_key(
        document_id: str, translation_id: str, fmt: str, pdf_mode: str | None = None
    ) -> str:
        return translation_file_key(document_id, translation_id, fmt, pdf_mode=pdf_mode)

    @staticmethod
    def digest_export_key(document_id: str, fmt: str = "docx") -> str:
        return export_key(document_id, f"digest.{fmt}")

    @staticmethod
    def summary_export_key(document_id: str, summary_id: str, fmt: str = "docx") -> str:
        return summary_export_key(document_id, summary_id, fmt)

    def _resolve_original_pdf(self, doc: Document) -> tuple[str | None, bool]:
        if doc.format != "pdf" or not doc.file_path:
            return None, False
        try:
            local = self.storage.resolve_local_or_key(doc.file_path)
            return local, local != doc.file_path
        except Exception:
            return None, False

    def _put_quality_manifest(self, key: str, quality) -> None:
        import json

        try:
            payload = quality.to_dict() if hasattr(quality, "to_dict") else dict(quality)
            self.storage.put_bytes(
                key,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                content_type="application/json",
            )
        except Exception:
            logger.debug("quality manifest write failed for %s", key, exc_info=True)

    def _build_reflow_pdf(
        self, content: str, *, title: str, headings: list[str] | None = None
    ) -> bytes:
        docx_bytes = build_docx_bytes_from_content(
            content, title=title, headings=headings, structured=True
        )
        return docx_bytes_to_pdf_bytes(docx_bytes)

    @staticmethod
    def _digest_download_name(title: str, fmt: str = "docx") -> str:
        safe_title = "".join(
            c if c.isalnum() or c in " -_" else "_" for c in (title or "document")
        )[:60]
        return f"digest_{safe_title}.{fmt}"

    @staticmethod
    def _summary_download_name(title: str, fmt: str = "docx") -> str:
        return f"summary_{safe_filename(title)}.{fmt}"

    # ── Invalidation ──────────────────────────────────────────────────

    def invalidate_document(self, document_id: str) -> None:
        self.storage.delete_prefix(document_prefix(document_id))

    def invalidate_ocr_exports(self, document_id: str) -> None:
        from utils.storage_keys import export_bg_prefix

        self.storage.delete_prefix(f"{document_prefix(document_id)}exports/")
        self.storage.delete_prefix(export_bg_prefix(document_id))

    def invalidate_digest_export(self, document_id: str) -> None:
        self.storage.delete(self.digest_export_key(document_id, "docx"))
        self.storage.delete(self.digest_export_key(document_id, "pdf"))

    def invalidate_summary_export(self, document_id: str, summary_id: str) -> None:
        self.storage.delete(self.summary_export_key(document_id, summary_id, "docx"))
        self.storage.delete(self.summary_export_key(document_id, summary_id, "pdf"))

    def mark_digest_dirty(self, document_id: str) -> None:
        """Drop cached digest when upstream sections change."""
        self.invalidate_digest_export(document_id)

    # ── OCR / normalized export build ─────────────────────────────────

    def build_ocr_export(
        self,
        db: Session,
        doc: Document,
        *,
        content_type: str = "ocr",
        mode: str = "auto",
        fmt: str = "docx",
        source: str = "auto",
        pdf_mode: str = "auto",
    ) -> Tuple[bytes, str, str]:
        """Return (bytes, download_filename, media_type)."""
        repo = DocumentRepository(db)
        use_original = source == "original" or (
            source == "auto" and is_native_word_document(doc.format)
        )
        base = f"{content_type}_{safe_filename(doc.title)}"

        if use_original:
            key = doc.file_path
            if not key or not self.storage.exists(key):
                raise FileNotFoundError("Original file not found in storage")
            if fmt == "pdf" and (doc.format or "").lower() == "pdf":
                data = self.storage.get_bytes(key)
                return (
                    data,
                    f"{os.path.splitext(doc.original_filename or base)[0]}.pdf",
                    "application/pdf",
                )
            if fmt == "pdf" and is_native_word_document(doc.format):
                local = self.storage.materialize_to_temp(key)
                try:
                    docx_bytes = open(local, "rb").read()
                    if (doc.format or "").lower() == "doc":
                        from services.extractors.doc_converter import convert_doc_to_docx

                        docx_path = convert_doc_to_docx(local)
                        docx_bytes = open(docx_path, "rb").read()
                    pdf_bytes = docx_bytes_to_pdf_bytes(docx_bytes)
                    pdf_name = f"{os.path.splitext(doc.original_filename or base)[0]}.pdf"
                    return pdf_bytes, pdf_name, "application/pdf"
                finally:
                    if os.path.isfile(local):
                        os.remove(local)
            download_name = doc.original_filename or os.path.basename(key)
            data = self.storage.get_bytes(key)
            media = (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if download_name.lower().endswith(".docx")
                else "application/octet-stream"
            )
            return data, download_name, media

        from utils.export_paths import resolve_ocr_content, spatial_export_plan

        dt = repo.get_digitized_text(doc.id)
        if not dt:
            raise ValueError("No extracted text found")
        content = resolve_ocr_content(repo, doc.id, dt, content_type=content_type)
        if not content:
            raise ValueError(f"No {content_type} content available")

        filename = f"{base}.docx"
        text_overridden = bool(getattr(dt, "text_overridden", False))
        use_spatial, embed_images = spatial_export_plan(
            repo,
            doc.id,
            mode=mode,
            text_overridden=text_overridden,
        )
        elements = []
        pages = []
        if use_spatial:
            elements = repo.get_elements(doc.id)
            pages = repo.get_pages(doc.id)

        structured_modes = mode not in ("plain", "markdown")
        if fmt == "pdf" and elements and pages and structured_modes:
            from utils.export_paths import resolve_ocr_pdf_mode
            from utils.storage_keys import ocr_quality_key

            requested = pdf_mode or "auto"
            effective = resolve_ocr_pdf_mode(requested, has_spatial=True)
            orig_path, orig_cleanup = self._resolve_original_pdf(doc)
            try:
                if effective == "reflow":
                    pdf_bytes = self._build_reflow_pdf(content, title=doc.title)
                    return pdf_bytes, f"{base}.reflow.pdf", "application/pdf"
                orig_bytes = None
                if orig_path:
                    try:
                        with open(orig_path, "rb") as fh:
                            orig_bytes = fh.read()
                    except OSError:
                        orig_bytes = None
                built = build_pdf_bytes_from_elements(
                    elements,
                    pages,
                    document_id=doc.id,
                    merge_blocks=True,
                    pdf_mode=effective,
                    text_kind="ocr",
                    original_pdf_bytes=orig_bytes,
                    original_pdf_path=None if orig_bytes else orig_path,
                    page_backgrounds=_resolve_export_page_backgrounds(doc, pages),
                )
                if isinstance(built, tuple):
                    pdf_bytes, render_result = built
                    used = render_result.pdf_mode
                    if requested == "auto" and not render_result.quality.ok:
                        pdf_bytes = self._build_reflow_pdf(content, title=doc.title)
                        used = "reflow"
                        render_result.quality.fallback = "reflow"
                        render_result.quality.pdf_mode = "reflow"
                    self._put_quality_manifest(
                        ocr_quality_key(doc.id, content_type=content_type, pdf_mode=used),
                        render_result.quality,
                    )
                    return pdf_bytes, f"{base}.{used}.pdf", "application/pdf"
                return built, f"{base}.pdf", "application/pdf"
            finally:
                if orig_cleanup and orig_path and os.path.isfile(orig_path):
                    os.remove(orig_path)

        if fmt == "docx" and structured_modes and use_spatial and elements:
            docx_bytes = build_docx_bytes_from_elements(
                elements,
                title=doc.title,
                document_id=doc.id,
                embed_images=embed_images,
            )
            return (
                docx_bytes,
                filename,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        structured = mode != "plain"
        if fmt == "pdf":
            docx_bytes = build_docx_bytes_from_content(
                content,
                title=doc.title,
                structured=structured,
            )
            pdf_bytes = docx_bytes_to_pdf_bytes(docx_bytes)
            return pdf_bytes, f"{base}.pdf", "application/pdf"

        docx_bytes = build_docx_bytes_from_content(
            content,
            title=doc.title,
            structured=structured,
        )
        return (
            docx_bytes,
            filename,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    def build_translation_export(
        self,
        db: Session,
        doc: Document,
        translation: Translation,
        *,
        source: str = "auto",
        fmt: str = "docx",
        pdf_mode: str = "auto",
    ) -> Tuple[bytes, str, str]:
        lang = translation.target_language.upper()
        base = f"translation_{lang}_{safe_filename(doc.title)}"
        filename = f"{base}.docx"

        use_structured = source in ("auto", "structured")

        if use_structured and translation.translation_mode == "pdf_overlay":
            key = translation.translated_file_path
            if not key or not self.storage.exists(key):
                raise FileNotFoundError("Translated PDF not found in storage")
            pdf_bytes = self.storage.get_bytes(key)
            if fmt == "pdf":
                return pdf_bytes, f"{base}.pdf", "application/pdf"
            # DOCX: reflow extracted text from the overlay PDF (same path as
            # OCR flat export). Spatial layout stays in the PDF download.
            content = (translation.translated_content or "").strip() or extract_pdf_text(pdf_bytes)
            if not content:
                raise ValueError("PDF overlay translation has no extractable text for DOCX")
            return self._translation_flat_export(
                doc,
                content,
                lang=lang,
                base=base,
                filename=filename,
                fmt="docx",
                structured=True,
            )

        if use_structured and translation.translation_mode == "docx_inplace":
            key = translation.translated_file_path
            if not key or not self.storage.exists(key):
                raise FileNotFoundError("Translated DOCX not found in storage")
            download_name = doc.original_filename or filename
            if not download_name.lower().endswith(".docx"):
                download_name = filename
            if fmt == "pdf":
                docx_bytes = self.storage.get_bytes(key)
                pdf_bytes = docx_bytes_to_pdf_bytes(docx_bytes)
                return pdf_bytes, f"{base}.pdf", "application/pdf"
            return (
                self.storage.get_bytes(key),
                download_name,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        from utils.export_paths import translation_spatial_plan

        if (
            use_structured
            and translation.translation_mode in ("element_based", "block_based")
            and translation.translated_elements
        ):
            elements = deserialize_translated_elements(translation.translated_elements)
            if elements:
                from utils.translation_elements import flatten_translated_elements

                repo = DocumentRepository(db)
                use_spatial, embed_images = translation_spatial_plan(
                    repo,
                    doc.id,
                    element_count=len(elements),
                    source=source,
                )
                if use_spatial:
                    if fmt == "pdf":
                        from utils.export_paths import resolve_translation_pdf_mode
                        from utils.storage_keys import translation_quality_key

                        pages = repo.get_pages(doc.id)
                        requested = pdf_mode or "auto"
                        effective = resolve_translation_pdf_mode(requested, has_spatial=True)
                        orig_path, orig_cleanup = self._resolve_original_pdf(doc)
                        try:
                            if effective == "reflow":
                                flat = (
                                    translation.translated_content
                                    or flatten_translated_elements(elements)
                                )
                                pdf_bytes = self._build_reflow_pdf(
                                    flat or "",
                                    title=doc.title,
                                    headings=[f"Translation ({lang})"],
                                )
                                return pdf_bytes, f"{base}.reflow.pdf", "application/pdf"
                            orig_bytes = None
                            if orig_path:
                                try:
                                    with open(orig_path, "rb") as fh:
                                        orig_bytes = fh.read()
                                except OSError:
                                    orig_bytes = None
                            built = build_pdf_bytes_from_elements(
                                elements,
                                pages,
                                document_id=doc.id,
                                merge_blocks=True,
                                pdf_mode=effective,
                                text_kind="translation",
                                original_pdf_bytes=orig_bytes,
                                original_pdf_path=None if orig_bytes else orig_path,
                                page_backgrounds=_resolve_export_page_backgrounds(doc, pages),
                                lang=(translation.target_language or "vi"),
                            )
                            if isinstance(built, tuple):
                                pdf_bytes, render_result = built
                                used = render_result.pdf_mode
                                if requested == "auto" and not render_result.quality.ok:
                                    from utils.translation_elements import (
                                        flatten_translated_elements,
                                    )

                                    flat = (
                                        translation.translated_content
                                        or flatten_translated_elements(elements)
                                    )
                                    pdf_bytes = self._build_reflow_pdf(
                                        flat or "",
                                        title=doc.title,
                                        headings=[f"Translation ({lang})"],
                                    )
                                    used = "reflow"
                                    render_result.quality.fallback = "reflow"
                                    render_result.quality.pdf_mode = "reflow"
                                self._put_quality_manifest(
                                    translation_quality_key(doc.id, translation.id, used),
                                    render_result.quality,
                                )
                                return pdf_bytes, f"{base}.{used}.pdf", "application/pdf"
                            return built, f"{base}.pdf", "application/pdf"
                        finally:
                            if orig_cleanup and orig_path and os.path.isfile(orig_path):
                                os.remove(orig_path)
                    docx_bytes = build_docx_bytes_from_elements(
                        elements,
                        title=doc.title,
                        document_id=doc.id,
                        embed_images=embed_images,
                    )
                    return (
                        docx_bytes,
                        filename,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                from utils.translation_elements import flatten_translated_elements

                flat = translation.translated_content or flatten_translated_elements(elements)
                if flat:
                    return self._translation_flat_export(
                        doc,
                        flat,
                        lang=lang,
                        base=base,
                        filename=filename,
                        fmt=fmt,
                        structured=source != "flat",
                    )

        if not translation.translated_content:
            raise ValueError("Translation has no content")

        return self._translation_flat_export(
            doc,
            translation.translated_content,
            lang=lang,
            base=base,
            filename=filename,
            fmt=fmt,
            structured=source != "flat",
        )

    @staticmethod
    def _translation_flat_export(
        doc: Document,
        content: str,
        *,
        lang: str,
        base: str,
        filename: str,
        fmt: str,
        structured: bool,
    ) -> Tuple[bytes, str, str]:
        if fmt == "pdf":
            docx_bytes = build_docx_bytes_from_content(
                content,
                title=doc.title,
                headings=[f"Translation ({lang})"],
                structured=structured,
            )
            pdf_bytes = docx_bytes_to_pdf_bytes(docx_bytes)
            return pdf_bytes, f"{base}.pdf", "application/pdf"

        docx_bytes = build_docx_bytes_from_content(
            content,
            title=doc.title,
            headings=[f"Translation ({lang})"],
            structured=structured,
        )
        return (
            docx_bytes,
            filename,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    def build_summary_export(
        self,
        db: Session,
        doc: Document,
        summary_id: str,
        content: str,
        fmt: str = "docx",
    ) -> Tuple[bytes, str, str]:
        docx_bytes = build_docx_bytes_from_content(
            content,
            title=doc.title,
            headings=["Summary"],
            structured=True,
        )
        filename = self._summary_download_name(doc.title, fmt)
        if fmt == "pdf":
            return (docx_bytes_to_pdf_bytes(docx_bytes), filename, media_for_fmt("pdf"))
        return (docx_bytes, filename, media_for_fmt("docx"))

    def build_digest_export(
        self, db: Session, document_id: str, fmt: str = "docx"
    ) -> Tuple[bytes, str, str]:
        digest = _digest_service.assemble(db, document_id)
        docx_bytes = _digest_renderer.render(digest)
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in digest.title)[:60]
        filename = f"digest_{safe_title}.{fmt}"
        if fmt == "pdf":
            return (docx_bytes_to_pdf_bytes(docx_bytes), filename, media_for_fmt("pdf"))
        return (docx_bytes, filename, media_for_fmt("docx"))

    # ── Cache helpers ─────────────────────────────────────────────────

    def put_export(self, key: str, data: bytes, *, content_type: str) -> str:
        return self.storage.put_bytes(key, data, content_type=content_type)

    def schedule_export_put(self, key: str, data: bytes, *, content_type: str) -> None:
        """Cache an export in MinIO after the client already has the bytes."""

        async def _bg() -> None:
            try:
                await asyncio.to_thread(self.put_export, key, data, content_type=content_type)
            except Exception:
                logger.exception("Background export put failed key=%s", key)

        try:
            asyncio.get_running_loop().create_task(_bg())
        except RuntimeError:
            # No event loop (sync tests / prefetch helpers) — put inline.
            self.put_export(key, data, content_type=content_type)

    def get_or_build_ocr_export(
        self,
        db: Session,
        doc: Document,
        *,
        content_type: str,
        mode: str,
        fmt: str,
        source: str,
        pdf_mode: str = "auto",
    ) -> Tuple[str, str, str, Optional[bytes]]:
        """Return (storage_key, download_filename, media_type, data_or_none).

        On cache hit ``data`` is None (stream from MinIO). On miss ``data`` is
        the built bytes — caller streams them immediately and may put in
        background; prefetch paths should call ``put_export`` synchronously.
        """
        if source in ("auto", "original") and is_native_word_document(doc.format) and fmt == "docx":
            key = doc.file_path
            if key and self.storage.exists(key):
                name = doc.original_filename or os.path.basename(key)
                media = mimetypes_guess(name)
                return key, name, media, None

        from utils.export_paths import resolve_ocr_pdf_mode, spatial_export_plan

        effective_mode = self._effective_ocr_mode(db, doc, mode, fmt=fmt)
        cache_mode = f"{effective_mode}__original" if source == "original" else effective_mode
        cache_pdf_mode = None
        if fmt == "pdf":
            repo = DocumentRepository(db)
            dt = repo.get_digitized_text(doc.id)
            text_overridden = bool(dt and getattr(dt, "text_overridden", False))
            use_spatial, _ = spatial_export_plan(
                repo, doc.id, mode=mode, text_overridden=text_overridden
            )
            cache_pdf_mode = resolve_ocr_pdf_mode(pdf_mode, has_spatial=use_spatial)
        key = self.ocr_export_key(
            doc.id,
            content_type=content_type,
            mode=cache_mode,
            fmt=fmt,
            pdf_mode=cache_pdf_mode,
        )
        if self.storage.exists(key):
            name = f"{content_type}_{safe_filename(doc.title)}.{fmt}"
            return key, name, media_for_fmt(fmt), None

        data, filename, media = self.build_ocr_export(
            db,
            doc,
            content_type=content_type,
            mode=mode,
            fmt=fmt,
            source=source,
            pdf_mode=pdf_mode,
        )
        return key, filename, media, data

    def get_or_build_translation_export(
        self,
        db: Session,
        doc: Document,
        translation: Translation,
        *,
        source: str,
        fmt: str,
        pdf_mode: str = "auto",
    ) -> Tuple[str, str, str, Optional[bytes]]:
        from utils.export_paths import resolve_translation_pdf_mode

        cache_pdf_mode = None
        if fmt == "pdf":
            has_spatial = translation.translation_mode in ("element_based", "block_based")
            cache_pdf_mode = resolve_translation_pdf_mode(pdf_mode, has_spatial=has_spatial)
            key = self.translation_export_key(doc.id, translation.id, fmt, pdf_mode=cache_pdf_mode)
            if not self.storage.exists(key) and (pdf_mode or "auto") == "auto":
                alt = self.translation_export_key(doc.id, translation.id, fmt, pdf_mode="reflow")
                if self.storage.exists(alt):
                    key = alt
        else:
            key = self.translation_export_key(doc.id, translation.id, fmt)
        if self.storage.exists(key):
            lang = translation.target_language.upper()
            base = f"translation_{lang}_{safe_filename(doc.title)}"
            return key, f"{base}.{fmt}", media_for_fmt(fmt), None

        data, filename, media = self.build_translation_export(
            db, doc, translation, source=source, fmt=fmt, pdf_mode=pdf_mode
        )
        used = cache_pdf_mode
        if fmt == "pdf":
            for token in ("reflow", "layout", "facsimile", "clean"):
                if filename.endswith(f".{token}.pdf"):
                    used = token
                    break
            key = self.translation_export_key(doc.id, translation.id, fmt, pdf_mode=used)
        return key, filename, media, data

    def get_or_build_digest_export(
        self, db: Session, document_id: str, fmt: str = "docx"
    ) -> Tuple[str, str, str, Optional[bytes]]:
        key = self.digest_export_key(document_id, fmt)
        doc = DocumentRepository(db).get(document_id)
        if not doc:
            raise ValueError("Document not found")
        download_name = self._digest_download_name(doc.title, fmt)

        if self.storage.exists(key):
            return key, download_name, media_for_fmt(fmt), None

        data, filename, media = self.build_digest_export(db, document_id, fmt)
        return key, filename, media, data

    def get_or_build_summary_export(
        self,
        db: Session,
        doc: Document,
        summary_id: str,
        content: str,
        fmt: str = "docx",
    ) -> Tuple[str, str, str, Optional[bytes]]:
        key = self.summary_export_key(doc.id, summary_id, fmt)
        download_name = self._summary_download_name(doc.title, fmt)
        if self.storage.exists(key):
            return key, download_name, media_for_fmt(fmt), None

        data, filename, media = self.build_summary_export(db, doc, summary_id, content, fmt)
        return key, download_name, media, data

    def export_cache_status(self, db: Session, document_id: str) -> dict:
        """Which cached download objects exist in MinIO."""
        doc = DocumentRepository(db).get(document_id)
        if not doc:
            return {}
        ocr_mode_docx = self._effective_ocr_mode(db, doc, "auto", fmt="docx")
        ocr_mode_pdf = self._effective_ocr_mode(db, doc, "auto", fmt="pdf")
        from utils.export_paths import resolve_ocr_pdf_mode, resolve_translation_pdf_mode

        ocr_pdf_mode = resolve_ocr_pdf_mode(
            "auto", has_spatial=ocr_mode_pdf in ("layout", "spatial")
        )
        status = {
            "ocr_docx": self.storage.exists(
                self.ocr_export_key(document_id, content_type="ocr", mode=ocr_mode_docx, fmt="docx")
            ),
            "ocr_pdf": self.storage.exists(
                self.ocr_export_key(
                    document_id,
                    content_type="ocr",
                    mode=ocr_mode_pdf,
                    fmt="pdf",
                    pdf_mode=ocr_pdf_mode,
                )
            )
            or self.storage.exists(
                self.ocr_export_key(document_id, content_type="ocr", mode=ocr_mode_pdf, fmt="pdf")
            ),
            "ocr_pdf_mode": ocr_pdf_mode,
            "normalized_docx": self.storage.exists(
                self.ocr_export_key(
                    document_id, content_type="normalized", mode=ocr_mode_docx, fmt="docx"
                )
            ),
            "normalized_pdf": self.storage.exists(
                self.ocr_export_key(
                    document_id,
                    content_type="normalized",
                    mode=ocr_mode_pdf,
                    fmt="pdf",
                    pdf_mode=ocr_pdf_mode,
                )
            ),
            "digest_docx": self.storage.exists(self.digest_export_key(document_id)),
        }
        from data.db_models import Summary, Translation

        trans = (
            db.query(Translation)
            .filter(Translation.document_id == document_id, Translation.status == "COMPLETED")
            .order_by(Translation.created_at.desc())
            .first()
        )
        if trans:
            trans_pdf_mode = resolve_translation_pdf_mode(
                "auto",
                has_spatial=trans.translation_mode in ("element_based", "block_based"),
            )
            status["translation_docx"] = self.storage.exists(
                self.translation_export_key(document_id, trans.id, "docx")
            )
            status["translation_pdf"] = self.storage.exists(
                self.translation_export_key(document_id, trans.id, "pdf", pdf_mode=trans_pdf_mode)
            ) or self.storage.exists(self.translation_export_key(document_id, trans.id, "pdf"))
            status["translation_pdf_mode"] = trans_pdf_mode
            status["translation_mode"] = trans.translation_mode
        summary = (
            db.query(Summary)
            .filter(Summary.document_id == document_id, Summary.status == "COMPLETED")
            .order_by(Summary.created_at.desc())
            .first()
        )
        if summary:
            status["summary_docx"] = self.storage.exists(
                self.summary_export_key(document_id, summary.id)
            )
        return status

    def _effective_ocr_mode(
        self, db: Session, doc: Document, mode: str, *, fmt: str = "docx"
    ) -> str:
        if mode in ("plain", "markdown"):
            return mode
        from utils.export_paths import spatial_export_plan

        repo = DocumentRepository(db)
        dt = repo.get_digitized_text(doc.id)
        text_overridden = bool(dt and getattr(dt, "text_overridden", False))
        use_spatial, _ = spatial_export_plan(
            repo,
            doc.id,
            mode=mode,
            text_overridden=text_overridden,
        )
        if use_spatial:
            return "layout" if fmt == "pdf" else "spatial"
        return "markdown"

    async def cache_ocr_exports_after_extract(self, db: Session, document_id: str) -> None:
        """Pre-build OCR/normalized DOCX + facsimile PDF exports during task dead time."""
        doc = DocumentRepository(db).get(document_id)
        if not doc:
            return
        if is_native_word_document(doc.format):
            return

        async def _one(content_type: str, mode: str, fmt: str, *, pdf_mode: str = "auto") -> None:
            dbm = get_db_manager()
            with dbm.session() as session:
                doc_row = DocumentRepository(session).get(document_id)
                if not doc_row:
                    return
                try:
                    key, _name, media, data = await asyncio.to_thread(
                        self.get_or_build_ocr_export,
                        session,
                        doc_row,
                        content_type=content_type,
                        mode=mode,
                        fmt=fmt,
                        source="extracted",
                        pdf_mode=pdf_mode,
                    )
                    if data is not None:
                        await asyncio.to_thread(
                            self.put_export, key, data, content_type=media
                        )
                except Exception as exc:
                    logger.warning(
                        "OCR export cache failed %s/%s/%s: %s",
                        content_type,
                        mode,
                        fmt,
                        exc,
                    )

        for content_type in ("ocr", "normalized"):
            await _one(content_type, "auto", "docx")
            await _one(content_type, "auto", "pdf", pdf_mode="facsimile")

    async def cache_translation_exports(
        self,
        db: Session,
        document_id: str,
        translation_id: str,
    ) -> None:
        async def _one(fmt: str, *, pdf_mode: str = "auto") -> None:
            dbm = get_db_manager()
            with dbm.session() as session:
                doc = DocumentRepository(session).get(document_id)
                trans = TranslationRepository(session).get(translation_id, document_id)
                if not doc or not trans:
                    return
                try:
                    key, _name, media, data = await asyncio.to_thread(
                        self.get_or_build_translation_export,
                        session,
                        doc,
                        trans,
                        source="auto",
                        fmt=fmt,
                        pdf_mode=pdf_mode,
                    )
                    if data is not None:
                        await asyncio.to_thread(
                            self.put_export, key, data, content_type=media
                        )
                except Exception as exc:
                    logger.warning("Translation export cache failed %s: %s", fmt, exc)

        await _one("docx")
        await _one("pdf", pdf_mode="layout")

    async def cache_digest_export(self, document_id: str) -> None:
        dbm = get_db_manager()
        with dbm.session() as session:
            self.mark_digest_dirty(document_id)
        with dbm.session() as session:
            try:
                key, _name, media, data = await asyncio.to_thread(
                    self.get_or_build_digest_export,
                    session,
                    document_id,
                )
                if data is not None:
                    await asyncio.to_thread(self.put_export, key, data, content_type=media)
            except Exception as exc:
                logger.warning("Digest export cache failed: %s", exc)

    async def cache_summary_export(self, document_id: str, summary_id: str) -> None:
        dbm = get_db_manager()
        with dbm.session() as session:
            doc = DocumentRepository(session).get(document_id)
            from data.db_models import Summary

            summary = (
                session.query(Summary)
                .filter(Summary.id == summary_id, Summary.document_id == document_id)
                .first()
            )
            if not doc or not summary or not summary.content:
                return
            try:
                key, _name, media, data = await asyncio.to_thread(
                    self.get_or_build_summary_export,
                    session,
                    doc,
                    summary_id,
                    summary.content,
                )
                if data is not None:
                    await asyncio.to_thread(self.put_export, key, data, content_type=media)
            except Exception as exc:
                logger.warning("Summary export cache failed: %s", exc)


def media_for_fmt(fmt: str) -> str:
    if fmt == "pdf":
        return "application/pdf"
    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def mimetypes_guess(filename: str) -> str:
    import mimetypes

    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


export_service = ExportService()
