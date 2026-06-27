"""Build and cache document exports in MinIO."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Tuple

from sqlalchemy.orm import Session

from config.settings import settings
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
    is_native_word_document,
    safe_filename,
)
from utils.storage_keys import (
    document_prefix,
    export_key,
    ocr_export_name,
    translation_file_key,
)
from utils.translation_elements import deserialize_translated_elements

logger = logging.getLogger(__name__)

_digest_renderer = DigestRenderer()
_digest_service = DigestService()


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

  # ── Invalidation ──────────────────────────────────────────────────

    def invalidate_document(self, document_id: str) -> None:
        self.storage.delete_prefix(document_prefix(document_id))

    def invalidate_ocr_exports(self, document_id: str) -> None:
        self.storage.delete_prefix(f"{document_prefix(document_id)}exports/")

    def invalidate_digest_export(self, document_id: str) -> None:
        self.storage.delete(self.digest_export_key(document_id))

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

        from utils.content_storage import read_text_field
        from utils.text_assembly import assemble_ocr_from_pages

        dt = repo.get_digitized_text(doc.id)
        if not dt:
            raise ValueError("No extracted text found")
        if content_type == "ocr":
            pages = repo.get_pages(doc.id)
            content = (
                assemble_ocr_from_pages(pages)
                if pages
                else read_text_field(
                    inline=dt.ocr_content,
                    key=dt.ocr_content_key,
                )
            )
        else:
            content = read_text_field(
                inline=dt.normalized_content,
                key=dt.normalized_content_key,
            )
        if not content:
            raise ValueError(f"No {content_type} content available")

        filename = f"{base}.docx"
        spatial_cap = settings.ocr_download_spatial_max_elements
        text_overridden = bool(getattr(dt, "text_overridden", False))
        elements = []
        pages = []
        if not text_overridden:
            element_count = repo.count_elements(doc.id)
            if element_count > 0 and element_count <= spatial_cap:
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
            )
            return pdf_bytes, f"{base}.pdf", "application/pdf"

        if (
            fmt == "docx"
            and structured_modes
            and not text_overridden
            and elements
        ):
            docx_bytes = build_docx_bytes_from_elements(
                elements, title=doc.title, document_id=doc.id
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

        if (
            use_structured
            and translation.translation_mode in ("element_based", "block_based")
            and translation.translated_elements
        ):
            elements = deserialize_translated_elements(translation.translated_elements)
            if elements:
                if fmt == "pdf":
                    pages = DocumentRepository(db).get_pages(doc.id)
                    pdf_bytes = build_pdf_bytes_from_elements(
                        elements,
                        pages,
                        document_id=doc.id,
                        merge_blocks=True,
                        text_overlay="replace",
                    )
                    return pdf_bytes, f"{base}.pdf", "application/pdf"
                docx_bytes = build_docx_bytes_from_elements(
                    elements, title=doc.title, document_id=doc.id
                )
                return (
                    docx_bytes,
                    filename,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

        if not translation.translated_content:
            raise ValueError("Translation has no content")

        if fmt == "pdf":
            docx_bytes = build_docx_bytes_from_content(
                translation.translated_content,
                title=doc.title,
                headings=[f"Translation ({lang})"],
                structured=source != "flat",
            )
            pdf_bytes = docx_bytes_to_pdf_bytes(docx_bytes)
            return pdf_bytes, f"{base}.pdf", "application/pdf"

        docx_bytes = build_docx_bytes_from_content(
            translation.translated_content,
            title=doc.title,
            headings=[f"Translation ({lang})"],
            structured=source != "flat",
        )
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

        # Determine effective mode for cache key
        effective_mode = self._effective_ocr_mode(db, doc, mode, fmt=fmt)
        key = self.ocr_export_key(
            doc.id, content_type=content_type, mode=effective_mode, fmt=fmt
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
        if self.storage.exists(key):
            digest = _digest_service.assemble(db, document_id)
            safe_title = "".join(
                c if c.isalnum() or c in " -_" else "_" for c in digest.title
            )[:60]
            return key, f"digest_{safe_title}.docx", media_for_fmt("docx")

        data, filename, media = self.build_digest_export(db, document_id)
        self.put_export(key, data, content_type=media)
        return key, filename, media

    def _effective_ocr_mode(self, db: Session, doc: Document, mode: str, *, fmt: str = "docx") -> str:
        if mode in ("plain", "markdown"):
            return mode
        repo = DocumentRepository(db)
        dt = repo.get_digitized_text(doc.id)
        text_overridden = bool(dt and getattr(dt, "text_overridden", False))
        if text_overridden:
            return "markdown"
        cap = settings.ocr_download_spatial_max_elements
        count = repo.count_elements(doc.id)
        if count > 0 and count <= cap:
            return "layout" if fmt == "pdf" else "spatial"
        return "markdown"

    async def cache_ocr_exports_after_extract(self, db: Session, document_id: str) -> None:
        doc = DocumentRepository(db).get(document_id)
        if not doc:
            return
        if is_native_word_document(doc.format):
            return

        async def _one(content_type: str, mode: str, fmt: str) -> None:
            try:
                await asyncio.to_thread(
                    self.get_or_build_ocr_export,
                    db,
                    doc,
                    content_type=content_type,
                    mode=mode,
                    fmt=fmt,
                    source="extracted",
                )
            except Exception as exc:
                logger.warning("OCR export cache failed %s/%s/%s: %s", content_type, mode, fmt, exc)

        for content_type in ("ocr", "normalized"):
            for fmt in ("docx", "pdf"):
                await _one(content_type, "auto", fmt)

    async def cache_translation_exports(
        self,
        db: Session,
        document_id: str,
        translation_id: str,
    ) -> None:
        doc = DocumentRepository(db).get(document_id)
        trans = TranslationRepository(db).get(translation_id, document_id)
        if not doc or not trans:
            return

        async def _one(fmt: str) -> None:
            try:
                if trans.translation_mode == "pdf_overlay" and fmt == "docx":
                    return
                await asyncio.to_thread(
                    self.get_or_build_translation_export,
                    db,
                    doc,
                    trans,
                    source="auto",
                    fmt=fmt,
                )
            except Exception as exc:
                logger.warning("Translation export cache failed %s: %s", fmt, exc)

        await _one("docx")
        await _one("pdf")


def media_for_fmt(fmt: str) -> str:
    if fmt == "pdf":
        return "application/pdf"
    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def mimetypes_guess(filename: str) -> str:
    import mimetypes

    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


export_service = ExportService()
