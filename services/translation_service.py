"""
Translation service.

Structure-preserving translation:
- DOCX/DOC  → in-place file translation (preserves layout)
- PDF/image → per layout-element translation (preserves bbox/reading order)
- Fallback  → tree index, then flat chunk translation
"""
import os

from config.settings import normalize_lang_code, settings
from data.database import get_db_manager
from data.db_models import Translation, TreeIndex, DigitizedText
from services.base_service import BaseTaskService
from services.task_manager import task_manager
from utils.translation_elements import serialize_translated_elements


class TranslationService(BaseTaskService):
    """Document translation service (background task)."""

    def submit(self, db, document_id: str, target_language: str = "vi", domain: str = "general") -> tuple:
        """Create a translation record and submit background task.

        Returns (task_id, translation_id, reused).
        """
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        doc = repo.get(document_id)
        if not doc:
            raise ValueError("Document not found")

        target_language = normalize_lang_code(target_language)
        source_language = normalize_lang_code(doc.source_language or "en")
        if target_language == source_language:
            raise ValueError(
                f"Target language must differ from source ({source_language})"
            )

        existing_task = task_manager.get_active_task_id(db, document_id, "TRANSLATE")
        matching_trans = (
            db.query(Translation)
            .filter(
                Translation.document_id == document_id,
                Translation.target_language == target_language,
                Translation.status.in_(["PENDING", "IN_PROGRESS"]),
            )
            .order_by(Translation.created_at.desc())
            .first()
        )
        if existing_task and matching_trans:
            return existing_task, matching_trans.id, True

        trans = Translation(
            document_id=document_id,
            target_language=target_language,
            status="PENDING",
        )
        db.add(trans)
        db.commit()
        db.refresh(trans)
        translation_id = trans.id

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="TRANSLATE",
            coro_factory=lambda tid: self._translate(
                document_id, target_language, domain, translation_id, tid
            ),
            dedupe=not (existing_task and not matching_trans),
        )
        return task_id, translation_id, False

    async def _translate(
        self,
        document_id: str,
        target_language: str,
        domain: str = "general",
        translation_id: str = None,
        task_id: str = None,
    ):
        db_manager = get_db_manager()

        def _set_status(status: str):
            if not translation_id:
                return
            with db_manager.session() as db:
                t = db.query(Translation).filter(Translation.id == translation_id).first()
                if t:
                    t.status = status

        _set_status("IN_PROGRESS")

        try:
            await self._wait_for_digitized_text(document_id, task_id=task_id)

            with db_manager.session() as db:
                from data.repositories import DocumentRepository
                repo = DocumentRepository(db)
                dt = repo.get_digitized_text(document_id)
                if not dt:
                    raise ValueError("No digitized text — run OCR first")
                doc = repo.get(document_id)
                if not doc:
                    raise ValueError("Document not found")
                source_lang = normalize_lang_code(doc.source_language or "en")
                doc_format = (doc.format or "").lower()
                file_path = doc.file_path
                flat_text = dt.normalized_content or dt.ocr_content or ""

            from api.dependencies import get_llm_client
            from core.pageindex.enrichment.translator import StructuredTranslator
            from services.translators import (
                DocxInPlaceTranslator,
                PdfOverlayTranslator,
            )

            llm_client = get_llm_client()
            translator = StructuredTranslator(
                llm_client=llm_client,
                source_lang=source_lang,
                target_lang=target_language,
                domain=domain,
                chunk_size=settings.ai_chunk_tokens,
            )

            async def on_progress(pct: int, msg: str):
                self._progress(task_id, pct, msg)

            if doc_format in ("docx", "doc"):
                if not file_path:
                    raise ValueError("Original file path missing for DOCX translation")
                out_dir = os.path.join(settings.upload_dir, "translations")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{translation_id}.docx")
                result = await DocxInPlaceTranslator(translator).translate_file(
                    file_path,
                    out_path,
                    doc_format=doc_format,
                    on_progress=on_progress,
                )
            elif (
                doc_format == "pdf"
                and settings.enable_pdf_overlay
                and file_path
            ):
                from data.repositories import DocumentRepository

                with db_manager.session() as db:
                    scanned = DocumentRepository(db).count_scanned_pages(document_id)

                if scanned is None:
                    from services.extractors.pdf_text_extractor import classify_pages

                    page_types = classify_pages(file_path, threshold=settings.pdf_text_threshold)
                    scanned = sum(1 for v in page_types.values() if v == "scanned")

                if scanned == 0:
                    out_dir = os.path.join(settings.upload_dir, "translations")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{translation_id}.pdf")
                    try:
                        result = await PdfOverlayTranslator().translate_file(
                            file_path,
                            out_path,
                            source_lang=source_lang,
                            target_lang=target_language,
                            on_progress=on_progress,
                        )
                    except Exception:
                        result = await self._translate_pdf_elements_or_flat(
                            document_id,
                            flat_text,
                            translator,
                            on_progress=on_progress,
                        )
                else:
                    result = await self._translate_pdf_elements_or_flat(
                        document_id,
                        flat_text,
                        translator,
                        on_progress=on_progress,
                    )
            else:
                result = await self._translate_pdf_elements_or_flat(
                    document_id,
                    flat_text,
                    translator,
                    on_progress=on_progress,
                )

            self._progress(task_id, 98, "Saving translation")

            with db_manager.session() as db:
                if translation_id:
                    t = db.query(Translation).filter(Translation.id == translation_id).first()
                    if t:
                        t.translated_content = result.get("translated_content")
                        t.translated_file_path = result.get("translated_file_path")
                        elements = result.get("translated_elements")
                        t.translated_elements = (
                            serialize_translated_elements(elements) if elements else None
                        )
                        t.translation_mode = result.get("translation_mode")
                        t.status = "COMPLETED"

            self._progress(task_id, 100, "Done")
            return {
                "translation_length": len(result.get("translated_content") or ""),
                "target_language": target_language,
                "translation_mode": result.get("translation_mode"),
            }
        except Exception:
            _set_status("FAILED")
            raise

    async def _translate_pdf_elements_or_flat(
        self,
        document_id: str,
        flat_text: str,
        translator,
        *,
        on_progress,
    ) -> dict:
        """Element-based, tree, or flat translation for scanned/mixed PDFs."""
        from utils.translation_elements import layout_element_to_dict

        db_manager = get_db_manager()
        element_payloads: list[dict] = []
        tree_data = None
        text_overridden = False
        with db_manager.session() as db:
            from sqlalchemy.orm import joinedload
            from data.db_models import LayoutElement, Page

            dt = (
                db.query(DigitizedText)
                .filter(DigitizedText.document_id == document_id)
                .first()
            )
            text_overridden = bool(dt and getattr(dt, "text_overridden", False))

            elements = []
            if not text_overridden:
                from data.repositories import DocumentRepository

                doc_repo = DocumentRepository(db)
                cap = settings.ocr_download_spatial_max_elements
                element_count = doc_repo.count_elements(document_id)
                if element_count > 0 and element_count <= cap:
                    elements = (
                        db.query(LayoutElement)
                        .join(Page)
                        .filter(Page.document_id == document_id)
                        .options(joinedload(LayoutElement.page))
                        .order_by(Page.page_number, LayoutElement.sequence_order)
                        .all()
                    )
            tree_index = (
                db.query(TreeIndex)
                .filter(TreeIndex.document_id == document_id)
                .order_by(TreeIndex.created_at.desc())
                .first()
            )
            if tree_index:
                tree_data = tree_index.tree_data

            cap = settings.ocr_download_spatial_max_elements
            if elements and len(elements) <= cap and not text_overridden:
                element_payloads = [
                    layout_element_to_dict(
                        elem,
                        elem.page.page_number if elem.page else 1,
                    )
                    for elem in elements
                ]

        from services.translators import ElementTranslator, FlatTranslator, TreeTranslator

        if element_payloads:
            return await ElementTranslator(translator).translate_payloads(
                element_payloads,
                on_progress=on_progress,
            )
        if tree_data and not text_overridden:
            return await TreeTranslator(translator).translate_tree(
                tree_data,
                on_progress=on_progress,
            )
        if flat_text:
            return await FlatTranslator(translator).translate_text(
                flat_text,
                on_progress=on_progress,
            )
        raise ValueError("No text content available")
