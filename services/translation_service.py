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
from data.db_models import Translation, TreeIndex
from services.base_service import BaseTaskService
from services.task_manager import task_manager
from utils.translation_elements import serialize_translated_elements


class TranslationService(BaseTaskService):
    """Document translation service (background task)."""

    def submit(self, db, document_id: str, target_language: str = "vi", domain: str = "general") -> tuple:
        """Create a translation record and submit background task.

        Returns (task_id, translation_id).
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
            coro=self._translate(document_id, target_language, domain, translation_id),
        )
        return task_id, translation_id

    async def _translate(
        self,
        document_id: str,
        target_language: str,
        domain: str = "general",
        translation_id: str = None,
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

            task_id = self._find_task_id(document_id, "TRANSLATE")

            from api.dependencies import get_llm_client
            from core.pageindex.enrichment.translator import StructuredTranslator
            from services.translators import (
                DocxInPlaceTranslator,
                ElementTranslator,
                FlatTranslator,
                TreeTranslator,
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
            else:
                with db_manager.session() as db:
                    from data.repositories import DocumentRepository
                    from sqlalchemy.orm import joinedload
                    from data.db_models import LayoutElement, Page

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
                    flat_text = dt.normalized_content or dt.ocr_content

                if elements:
                    result = await ElementTranslator(translator).translate_elements(
                        elements,
                        on_progress=on_progress,
                    )
                elif tree_index and tree_index.tree_data:
                    result = await TreeTranslator(translator).translate_tree(
                        tree_index.tree_data,
                        on_progress=on_progress,
                    )
                elif flat_text:
                    result = await FlatTranslator(translator).translate_text(
                        flat_text,
                        on_progress=on_progress,
                    )
                else:
                    raise ValueError("No text content available")

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
