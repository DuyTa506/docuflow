"""Extract bibliographic metadata (§1) from document front matter via LLM."""

import logging
from typing import Optional

from config.settings import settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from services.base_service import BaseTaskService
from services.task_manager import task_manager
from utils.digest_format import bibliographic_defaults

logger = logging.getLogger(__name__)


class BibliographicService(BaseTaskService):
    """Extract title, authors, publisher, ISBN, etc. from OCR text."""

    def submit(self, db, document_id: str) -> tuple[str, bool]:
        from data.repositories import DocumentRepository

        if not DocumentRepository(db).get(document_id):
            raise ValueError("Document not found")

        existing = task_manager.get_active_task_id(db, document_id, "BIBLIOGRAPHIC")
        if existing:
            return existing, True

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="BIBLIOGRAPHIC",
            coro_factory=lambda tid: self._extract(document_id, tid),
        )
        return task_id, False

    async def run_for_pipeline(self, document_id: str, task_id: Optional[str] = None):
        """Run bibliographic extraction inside Temporal pipeline (no task_manager)."""
        return await self._extract(document_id, task_id)

    def _read_front_matter(self, document_id: str, max_chars: int = 12000) -> str:
        text = self._read_text(document_id)
        return text[:max_chars]

    async def _extract(self, document_id: str, task_id: Optional[str] = None):
        db_manager = get_db_manager()
        self._progress(task_id, 10, "Reading front matter")

        with db_manager.session() as db:
            from data.repositories import DocumentRepository

            doc = DocumentRepository(db).get(document_id)
            if not doc:
                raise ValueError("Document not found")
            default_pages = doc.total_pages
            doc_title = doc.title or ""

        excerpt = self._read_front_matter(document_id)

        from api.dependencies import get_llm_client

        llm = get_llm_client()
        enricher = BaseEnricher(llm)
        from utils.prompt_budget import PromptBudget, PromptBudgetError, allocate_document_sample

        fixed_prefix = (
            "You are a library cataloging assistant.\n\n"
            "TASK: Extract bibliographic metadata from the document excerpt below.\n"
            "Return ONLY valid JSON with these keys (use empty string if unknown):\n"
            "{\n"
            '  "title_display": "The full title in the document own language first, then '
            "a Vietnamese translation in parentheses — e.g. `Advances in Adaptive Radar "
            "Detection (Những tiến bộ trong phát hiện radar thích ứng)`. This holds for "
            'any source language. For a Vietnamese document, the title alone",\n'
            '  "authors": "Comma-separated author names",\n'
            '  "publisher": "Publisher name",\n'
            '  "year": "Publication year",\n'
            '  "isbn": "ISBN if present",\n'
            '  "doi": "DOI if present",\n'
            '  "pages": "Page count as string"\n'
            "}\n\n"
            "RULES: Every value MUST be supported by the excerpt. Do NOT invent data.\n\n"
            "DOCUMENT EXCERPT:\n"
        )
        fixed_suffix = "\n\nJSON:"
        budget = PromptBudget(
            context_tokens=settings.ai_model_context_window,
            output_reserve=settings.ai_output_reserve_tokens,
        )
        try:
            excerpt, budget_meta = allocate_document_sample(
                document_id=document_id,
                text=excerpt,
                enricher=enricher,
                budget=budget,
                fixed_parts=[fixed_prefix, fixed_suffix],
                sample_builder=lambda sample_budget: enricher.truncate_to_tokens(
                    excerpt, sample_budget
                ),
            )
        except PromptBudgetError as exc:
            logger.error("Bibliographic prompt budget exceeded for %s: %s", document_id, exc)
            raise ValueError(f"Bibliographic prompt exceeds context window: {exc}") from exc

        prompt = f"{fixed_prefix}{excerpt}{fixed_suffix}"
        logger.info("bibliographic prompt budget document_id=%s meta=%s", document_id, budget_meta)

        self._progress(task_id, 50, "Extracting metadata")
        response = await llm.chat_completion(prompt)

        try:
            meta = llm.extract_json(response)
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}

        defaults = bibliographic_defaults(title=doc_title, pages=default_pages)
        for key in defaults:
            val = meta.get(key)
            if val is not None and str(val).strip():
                defaults[key] = str(val).strip()
        if not defaults["title_display"]:
            defaults["title_display"] = doc_title

        self._progress(task_id, 90, "Saving metadata")
        with db_manager.session() as db:
            from data.db_models import Document

            row = db.query(Document).filter(Document.id == document_id).first()
            if row:
                row.bibliographic_metadata = defaults

        self._progress(task_id, 100, "Done")
        return defaults
