"""
Translation service.

Translates document text to a target language using the existing
StructuredTranslator from pageindex/enrichment/translator.py.
Runs as a background task via TaskManager.
"""
from config.settings import settings
from data.database import get_db_manager
from data.db_models import Translation
from services.base_service import BaseTaskService
from services.task_manager import task_manager


class TranslationService(BaseTaskService):
    """Document translation service (background task)."""

    def submit(self, db, document_id: str, target_language: str = "vi", domain: str = "general") -> tuple:
        """Create a translation record and submit background task.

        Returns (task_id, translation_id).
        """
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")

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

    async def _translate(self, document_id: str, target_language: str, domain: str = "general", translation_id: str = None):
        db_manager = get_db_manager()

        def _set_status(status: str):
            if not translation_id:
                return
            with db_manager.session() as db:
                t = db.query(Translation).filter(Translation.id == translation_id).first()
                if t:
                    t.status = status

        # Mark IN_PROGRESS
        _set_status("IN_PROGRESS")

        try:
            # Load text + source language
            with db_manager.session() as db:
                from data.repositories import DocumentRepository
                repo = DocumentRepository(db)
                dt = repo.get_digitized_text(document_id)
                if not dt:
                    raise ValueError("No digitized text — run OCR first")
                text = dt.normalized_content or dt.ocr_content
                if not text:
                    raise ValueError("No text content available")
                doc = repo.get(document_id)
                source_lang = doc.source_language if doc else "en"

            task_id = self._find_task_id(document_id, "TRANSLATE")

            # Create LLM client
            from api.dependencies import get_llm_client
            llm_client = get_llm_client()

            # Use the existing StructuredTranslator for chunked translation
            from core.pageindex.enrichment.translator import StructuredTranslator
            translator = StructuredTranslator(
                llm_client=llm_client,
                source_lang=source_lang,
                target_lang=target_language,
                domain=domain,
                chunk_size=settings.ai_chunk_tokens,
            )

            # Chunk and translate with progress
            chunks = translator.chunk_text(text, max_tokens=translator.chunk_size)
            translated_parts = []
            for i, chunk in enumerate(chunks):
                translated = await translator.translate_text(chunk)
                translated_parts.append(translated)
                pct = int(((i + 1) / len(chunks)) * 95)
                self._progress(task_id, pct, f"Chunk {i+1}/{len(chunks)}")

            full_translation = " ".join(translated_parts)

            # Update existing record with result and mark COMPLETED
            with db_manager.session() as db:
                if translation_id:
                    t = db.query(Translation).filter(Translation.id == translation_id).first()
                    if t:
                        t.translated_content = full_translation
                        t.status = "COMPLETED"
                else:
                    db.add(Translation(
                        document_id=document_id,
                        target_language=target_language,
                        translated_content=full_translation,
                        status="COMPLETED",
                    ))

            return {"translation_length": len(full_translation), "target_language": target_language}
        except Exception:
            _set_status("FAILED")
            raise
