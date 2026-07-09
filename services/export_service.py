"""Build and cache document exports in MinIO."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Tuple

from sqlalchemy.orm import Session

from config.settings import settings
from data.db_models import Document, Translation
from data.database import get_db_manager
from data.repositories import DocumentRepository, TranslationRepository
from services.digest_renderer import DigestRenderer
from services.digest_service import DigestService
from services.object_storage import get_object_storage
from utils.file_download import (
    build_docx_bytes_from_content,
    build_docx_bytes_from_elements,
    build_pdf_bytes_from_elements,
    docx_bytes_to_pdf_bytes,
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
    Returns {} (never raises) when the original isn't a resolvable PDF."""
    if doc.format != "pdf" or not doc.file_path:
        return {}
    from utils.layout_pdf import render_export_backgrounds

    storage = get_object_storage()
    local_path = None
    cleanup = False
    try:
        local_path = storage.resolve_local_or_key(doc.file_path)
        cleanup = local_path != doc.file_path
        page_numbers = [
            (p.get("page_number") if isinstance(p, dict) else getattr(p, "page_number", None))
            for p in pages
        ]
        page_numbers = [pn for pn in page_numbers if pn]
        return render_export_backgrounds(local_path, page_numbers)
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
    def ocr_export_key(document_id: str, *, content_type: str, mode: str, fmt: str) -> str:
        name = ocr_export_name(content_type=content_type, mode=mode, fmt=fmt)
        return export_key(document_id, name)

    @staticmethod
    def translation_export_key(document_id: str, translation_id: str, fmt: str) -> str:
        return translation_file_key(document_id, translation_id, fmt)

    @staticmethod
    def digest_export_key(document_id: str) -> str:
        return export_key(document_id, "digest.docx")

    @staticmethod
    def summary_export_key(document_id: str, summary_id: str) -> str:
        return summary_export_key(document_id, summary_id)

    @staticmethod
    def _digest_download_name(title: str) -> str:
        safe_title = "".join(
            c if c.isalnum() or c in " -_" else "_" for c in (title or "document")
        )[:60]
        return f"digest_{safe_title}.docx"

    @staticmethod
    def _summary_download_name(title: str) -> str:
        return f"summary_{safe_filename(title)}.docx"

  # ── Invalidation ──────────────────────────────────────────────────

    def invalidate_document(self, document_id: str) -> None:
        self.storage.delete_prefix(document_prefix(document_id))

    def invalidate_ocr_exports(self, document_id: str) -> None:
        self.storage.delete_prefix(f"{document_prefix(document_id)}exports/")

    def invalidate_digest_export(self, document_id: str) -> None:
        self.storage.delete(self.digest_export_key(document_id))

    def invalidate_summary_export(self, document_id: str, summary_id: str) -> None:
        self.storage.delete(self.summary_export_key(document_id, summary_id))

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
                return data, f"{os.path.splitext(doc.original_filename or base)[0]}.pdf", "application/pdf"
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
            pdf_bytes = build_pdf_bytes_from_elements(
                elements,
                pages,
                document_id=doc.id,
                merge_blocks=True,
                text_overlay="skip",
                page_backgrounds=_resolve_export_page_backgrounds(doc, pages),
            )
            return pdf_bytes, f"{base}.pdf", "application/pdf"

        if (
            fmt == "docx"
            and structured_modes
            and use_spatial
            and elements
        ):
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
    ) -> Tuple[bytes, str, str]:
        lang = translation.target_language.upper()
        base = f"translation_{lang}_{safe_filename(doc.title)}"
        filename = f"{base}.docx"

        use_structured = source in ("auto", "structured")

        if use_structured and translation.translation_mode == "pdf_overlay":
            key = translation.translated_file_path
            if fmt != "pdf":
                raise ValueError("PDF overlay translation has no DOCX export")
            if not key or not self.storage.exists(key):
                raise FileNotFoundError("Translated PDF not found in storage")
            return self.storage.get_bytes(key), f"{base}.pdf", "application/pdf"

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
                repo = DocumentRepository(db)
                use_spatial, embed_images = translation_spatial_plan(
                    repo,
                    doc.id,
                    element_count=len(elements),
                    source=source,
                )
                if use_spatial:
                    if fmt == "pdf":
                        pages = repo.get_pages(doc.id)
                        pdf_bytes = build_pdf_bytes_from_elements(
                            elements,
                            pages,
                            document_id=doc.id,
                            merge_blocks=True,
                            text_overlay="replace",
                            page_backgrounds=_resolve_export_page_backgrounds(doc, pages),
                        )
                        return pdf_bytes, f"{base}.pdf", "application/pdf"
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

                flat = translation.translated_content or flatten_translated_elements(
                    elements
                )
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
    ) -> Tuple[bytes, str, str]:
        docx_bytes = build_docx_bytes_from_content(
            content,
            title=doc.title,
            headings=["Summary"],
            structured=True,
        )
        filename = self._summary_download_name(doc.title)
        return (
            docx_bytes,
            filename,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    def build_digest_export(self, db: Session, document_id: str) -> Tuple[bytes, str, str]:
        digest = _digest_service.assemble(db, document_id)
        docx_bytes = _digest_renderer.render(digest)
        safe_title = "".join(
            c if c.isalnum() or c in " -_" else "_" for c in digest.title
        )[:60]
        filename = f"digest_{safe_title}.docx"
        return (
            docx_bytes,
            filename,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

  # ── Cache helpers ─────────────────────────────────────────────────

    def put_export(self, key: str, data: bytes, *, content_type: str) -> str:
        return self.storage.put_bytes(key, data, content_type=content_type)

    def get_or_build_ocr_export(
        self,
        db: Session,
        doc: Document,
        *,
        content_type: str,
        mode: str,
        fmt: str,
        source: str,
    ) -> Tuple[str, str, str]:
        """Return (storage_key, download_filename, media_type)."""
        if source in ("auto", "original") and is_native_word_document(doc.format) and fmt == "docx":
            key = doc.file_path
            if key and self.storage.exists(key):
                name = doc.original_filename or os.path.basename(key)
                media = mimetypes_guess(name)
                return key, name, media

        # Determine effective mode for cache key. `source="original"` returns
        # the raw uploaded file untouched (see build_ocr_export's use_original
        # branch) — it must not share a cache key with the rebuilt/extracted
        # export, or one would shadow the other.
        effective_mode = self._effective_ocr_mode(db, doc, mode, fmt=fmt)
        cache_mode = f"{effective_mode}__original" if source == "original" else effective_mode
        key = self.ocr_export_key(
            doc.id, content_type=content_type, mode=cache_mode, fmt=fmt
        )
        if self.storage.exists(key):
            name = f"{content_type}_{safe_filename(doc.title)}.{fmt}"
            return key, name, media_for_fmt(fmt)

        data, filename, media = self.build_ocr_export(
            db,
            doc,
            content_type=content_type,
            mode=mode,
            fmt=fmt,
            source=source,
        )
        self.put_export(key, data, content_type=media)
        return key, filename, media

    def get_or_build_translation_export(
        self,
        db: Session,
        doc: Document,
        translation: Translation,
        *,
        source: str,
        fmt: str,
    ) -> Tuple[str, str, str]:
        key = self.translation_export_key(doc.id, translation.id, fmt)
        if self.storage.exists(key):
            lang = translation.target_language.upper()
            base = f"translation_{lang}_{safe_filename(doc.title)}"
            return key, f"{base}.{fmt}", media_for_fmt(fmt)

        data, filename, media = self.build_translation_export(
            db, doc, translation, source=source, fmt=fmt
        )
        self.put_export(key, data, content_type=media)
        return key, filename, media

    def get_or_build_digest_export(self, db: Session, document_id: str) -> Tuple[str, str, str]:
        key = self.digest_export_key(document_id)
        doc = DocumentRepository(db).get(document_id)
        if not doc:
            raise ValueError("Document not found")
        download_name = self._digest_download_name(doc.title)

        if self.storage.exists(key):
            return key, download_name, media_for_fmt("docx")

        data, filename, media = self.build_digest_export(db, document_id)
        self.put_export(key, data, content_type=media)
        return key, filename, media

    def get_or_build_summary_export(
        self,
        db: Session,
        doc: Document,
        summary_id: str,
        content: str,
    ) -> Tuple[str, str, str]:
        key = self.summary_export_key(doc.id, summary_id)
        download_name = self._summary_download_name(doc.title)
        if self.storage.exists(key):
            return key, download_name, media_for_fmt("docx")

        data, filename, media = self.build_summary_export(
            db, doc, summary_id, content
        )
        self.put_export(key, data, content_type=media)
        return key, download_name, media

    def export_cache_status(self, db: Session, document_id: str) -> dict[str, bool]:
        """Which cached download objects exist in MinIO."""
        doc = DocumentRepository(db).get(document_id)
        if not doc:
            return {}
        ocr_mode_docx = self._effective_ocr_mode(db, doc, "auto", fmt="docx")
        ocr_mode_pdf = self._effective_ocr_mode(db, doc, "auto", fmt="pdf")
        status = {
            "ocr_docx": self.storage.exists(
                self.ocr_export_key(document_id, content_type="ocr", mode=ocr_mode_docx, fmt="docx")
            ),
            "ocr_pdf": self.storage.exists(
                self.ocr_export_key(document_id, content_type="ocr", mode=ocr_mode_pdf, fmt="pdf")
            ),
            "normalized_docx": self.storage.exists(
                self.ocr_export_key(
                    document_id, content_type="normalized", mode=ocr_mode_docx, fmt="docx"
                )
            ),
            "normalized_pdf": self.storage.exists(
                self.ocr_export_key(
                    document_id, content_type="normalized", mode=ocr_mode_pdf, fmt="pdf"
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
            status["translation_docx"] = self.storage.exists(
                self.translation_export_key(document_id, trans.id, "docx")
            )
            status["translation_pdf"] = self.storage.exists(
                self.translation_export_key(document_id, trans.id, "pdf")
            )
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

    def _effective_ocr_mode(self, db: Session, doc: Document, mode: str, *, fmt: str = "docx") -> str:
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
        """Pre-build OCR/normalized DOCX + PDF exports during task dead time."""
        doc = DocumentRepository(db).get(document_id)
        if not doc:
            return
        if is_native_word_document(doc.format):
            return

        async def _one(content_type: str, mode: str, fmt: str) -> None:
            dbm = get_db_manager()
            with dbm.session() as session:
                doc_row = DocumentRepository(session).get(document_id)
                if not doc_row:
                    return
                try:
                    await asyncio.to_thread(
                        self.get_or_build_ocr_export,
                        session,
                        doc_row,
                        content_type=content_type,
                        mode=mode,
                        fmt=fmt,
                        source="extracted",
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
            await _one(content_type, "auto", "pdf")

    async def cache_translation_exports(
        self,
        db: Session,
        document_id: str,
        translation_id: str,
    ) -> None:
        async def _one(fmt: str) -> None:
            dbm = get_db_manager()
            with dbm.session() as session:
                doc = DocumentRepository(session).get(document_id)
                trans = TranslationRepository(session).get(translation_id, document_id)
                if not doc or not trans:
                    return
                try:
                    if trans.translation_mode == "pdf_overlay" and fmt == "docx":
                        return
                    await asyncio.to_thread(
                        self.get_or_build_translation_export,
                        session,
                        doc,
                        trans,
                        source="auto",
                        fmt=fmt,
                    )
                except Exception as exc:
                    logger.warning("Translation export cache failed %s: %s", fmt, exc)

        await _one("docx")
        await _one("pdf")

    async def cache_digest_export(self, document_id: str) -> None:
        dbm = get_db_manager()
        with dbm.session() as session:
            self.mark_digest_dirty(document_id)
        with dbm.session() as session:
            try:
                await asyncio.to_thread(
                    self.get_or_build_digest_export,
                    session,
                    document_id,
                )
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
                await asyncio.to_thread(
                    self.get_or_build_summary_export,
                    session,
                    doc,
                    summary_id,
                    summary.content,
                )
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
