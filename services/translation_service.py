"""
Translation service.

Structure-preserving translation:
- DOCX/DOC  → in-place file translation (preserves layout)
- PDF/image → per layout-element translation (preserves bbox/reading order)
- Fallback  → tree index, then flat chunk translation
"""
import os
import tempfile

from sqlalchemy.exc import IntegrityError

from config.settings import normalize_lang_code, settings
from data.database import get_db_manager
from data.db_models import Translation, TreeIndex, DigitizedText
from services.base_service import BaseTaskService
from services.task_manager import task_manager
from utils.storage_keys import translation_file_key
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
        in_flight = (
            db.query(Translation)
            .filter(
                Translation.document_id == document_id,
                Translation.target_language == target_language,
                Translation.status.in_(["PENDING", "IN_PROGRESS"]),
            )
            .order_by(Translation.created_at.desc())
            .first()
        )
        if existing_task and in_flight:
            return existing_task, in_flight.id, True

        existing_trans = (
            db.query(Translation)
            .filter(
                Translation.document_id == document_id,
                Translation.target_language == target_language,
            )
            .order_by(Translation.created_at.desc())
            .first()
        )

        if existing_trans:
            translation_id = self._reset_translation_for_retry(
                db, existing_trans, document_id
            )
        else:
            trans = Translation(
                document_id=document_id,
                target_language=target_language,
                status="PENDING",
            )
            db.add(trans)
            try:
                db.commit()
            except IntegrityError:
                # Lost a race with a concurrent submit() for the same
                # (document_id, target_language) — the unique constraint
                # caught it. Fall back to reusing the row that won.
                db.rollback()
                existing_trans = (
                    db.query(Translation)
                    .filter(
                        Translation.document_id == document_id,
                        Translation.target_language == target_language,
                    )
                    .order_by(Translation.created_at.desc())
                    .first()
                )
                translation_id = self._reset_translation_for_retry(
                    db, existing_trans, document_id
                )
            else:
                db.refresh(trans)
                translation_id = trans.id

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="TRANSLATE",
            coro_factory=lambda tid: self._translate(
                document_id, target_language, domain, translation_id, tid
            ),
            dedupe=not (existing_task and not in_flight),
        )
        return task_id, translation_id, bool(existing_trans)

    @staticmethod
    def _reset_translation_for_retry(db, existing_trans: Translation, document_id: str) -> str:
        """Reset an existing Translation row (incl. a previously FAILED one)
        back to PENDING for a retry, instead of inserting a new row."""
        translation_id = existing_trans.id
        existing_trans.status = "PENDING"
        existing_trans.translated_content = None
        existing_trans.translated_file_path = None
        existing_trans.translated_elements = None
        existing_trans.translation_mode = None
        db.commit()
        db.refresh(existing_trans)
        from services.export_service import export_service

        for ext in ("docx", "pdf"):
            export_service.storage.delete(
                translation_file_key(document_id, translation_id, ext)
            )
        return translation_id

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
                from utils.content_storage import read_text_field

                flat_text = read_text_field(
                    inline=dt.normalized_content,
                    key=dt.normalized_content_key,
                )
                if not flat_text:
                    flat_text = read_text_field(
                        inline=dt.ocr_content,
                        key=dt.ocr_content_key,
                    )

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
                chunk_size=settings.ai_input_budget_tokens,
            )

            async def on_progress(pct: int, msg: str):
                self._progress(task_id, pct, msg)

            from services.object_storage import get_object_storage

            storage = get_object_storage()
            local_source = None
            source_cleanup = False
            needs_source_file = doc_format in ("docx", "doc") or (
                doc_format == "pdf" and settings.enable_pdf_overlay and file_path
            )
            if needs_source_file and file_path:
                local_source = storage.resolve_local_or_key(file_path)
                source_cleanup = local_source != file_path

            try:
                if doc_format in ("docx", "doc"):
                    if not local_source:
                        raise ValueError("Original file path missing for DOCX translation")
                    fd, tmp_out = tempfile.mkstemp(suffix=".docx")
                    os.close(fd)
                    try:
                        result = await DocxInPlaceTranslator(translator).translate_file(
                            local_source,
                            tmp_out,
                            doc_format=doc_format,
                            on_progress=on_progress,
                        )
                        object_key = translation_file_key(document_id, translation_id, "docx")
                        storage.put_file(object_key, tmp_out)
                        result["translated_file_path"] = object_key
                    finally:
                        if os.path.isfile(tmp_out):
                            os.remove(tmp_out)
                elif (
                    doc_format == "pdf"
                    and settings.enable_pdf_overlay
                    and local_source
                ):
                    from data.repositories import DocumentRepository

                    with db_manager.session() as db:
                        scanned = DocumentRepository(db).count_scanned_pages(document_id)

                    if scanned is None:
                        from services.extractors.pdf_text_extractor import classify_pages

                        page_types = classify_pages(
                            local_source, threshold=settings.pdf_text_threshold
                        )
                        scanned = sum(1 for v in page_types.values() if v == "scanned")

                    if scanned == 0:
                        fd, tmp_out = tempfile.mkstemp(suffix=".pdf")
                        os.close(fd)
                        try:
                            result = await PdfOverlayTranslator().translate_file(
                                local_source,
                                tmp_out,
                                source_lang=source_lang,
                                target_lang=target_language,
                                document_id=document_id,
                                on_progress=on_progress,
                            )
                            object_key = translation_file_key(document_id, translation_id, "pdf")
                            storage.put_file(object_key, tmp_out)
                            result["translated_file_path"] = object_key
                        except Exception as overlay_exc:
                            import logging
                            logging.getLogger(__name__).warning(
                                "PDF overlay failed, falling back to block/element: %s",
                                overlay_exc,
                                exc_info=True,
                            )
                            result = await self._translate_pdf_elements_or_flat(
                                document_id,
                                flat_text,
                                translator,
                                on_progress=on_progress,
                            )
                        finally:
                            if os.path.isfile(tmp_out):
                                os.remove(tmp_out)
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
            finally:
                if source_cleanup and local_source and os.path.isfile(local_source):
                    try:
                        os.remove(local_source)
                    except OSError:
                        pass

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
                        if elements and not result.get("translated_content"):
                            from utils.translation_elements import flatten_translated_elements

                            t.translated_content = flatten_translated_elements(elements)
                        t.translation_mode = result.get("translation_mode")
                        t.status = "IN_PROGRESS"

            self._progress(task_id, 99, "Preparing DOCX & PDF exports…")

            with db_manager.session() as db:
                from services.export_service import export_service

                await export_service.cache_translation_exports(
                    db, document_id, translation_id
                )

            with db_manager.session() as db:
                if translation_id:
                    t = db.query(Translation).filter(Translation.id == translation_id).first()
                    if t:
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
                from utils.tree_payload import get_tree_payload

                tree_data = get_tree_payload(db, tree_index)

            cap = settings.ocr_download_spatial_max_elements
            if elements and len(elements) <= cap and not text_overridden:
                element_payloads = [
                    layout_element_to_dict(
                        elem,
                        elem.page.page_number if elem.page else 1,
                    )
                    for elem in elements
                ]

        from services.translators import (
            BlockTranslator,
            ElementTranslator,
            FlatTranslator,
            TreeTranslator,
        )

        if element_payloads:
            count = len(element_payloads)
            use_blocks = (
                settings.translation_block_merge
                and count > settings.translation_element_max
            )
            if use_blocks:
                return await BlockTranslator(translator).translate_payloads(
                    element_payloads,
                    on_progress=on_progress,
                )
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
