"""Extract bibliographic metadata (§1) from document front matter via LLM."""

import logging
import re
from typing import Optional

from config.settings import settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from services.base_service import BaseTaskService
from services.domain_context import use_domain_context
from services.task_manager import task_manager
from utils.digest_format import bibliographic_defaults, format_title_display

logger = logging.getLogger(__name__)

# The imprint/credits page: authors, publisher, ISBN. A long table of contents
# can push it past the head window (DOC_014: at 15.9k chars).
_IMPRINT_RE = re.compile(
    r"^#*\s*(?:Credits|Authors?|Copyright|Выходные данные)\s*$|ISBN|Published by|©",
    re.M | re.I,
)
IMPRINT_SEARCH_CHARS = 60000
IMPRINT_EXCERPT_CHARS = 3000


def front_matter_excerpt(text: str, max_chars: int = 12000) -> str:
    """The first ``max_chars`` plus the imprint page when it lies further in."""
    head = text[:max_chars]
    imprint = _IMPRINT_RE.search(text, max_chars, IMPRINT_SEARCH_CHARS)
    if not imprint:
        return head
    start = text.rfind("\n", max_chars, imprint.start()) + 1 or imprint.start()
    return f"{head}\n\n[...]\n\n{text[start : start + IMPRINT_EXCERPT_CHARS]}"


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
        return front_matter_excerpt(self._read_text(document_id), max_chars)

    async def _extract(self, document_id: str, task_id: Optional[str] = None):
        db_manager = get_db_manager()
        self._progress(task_id, 10, "Đang đọc phần đầu tài liệu")

        with db_manager.session() as db:
            from data.repositories import DocumentRepository

            doc = DocumentRepository(db).get(document_id)
            if not doc:
                raise ValueError("Document not found")
            default_pages = doc.total_pages
            doc_title = doc.title or ""
            source_language = doc.source_language or ""

        excerpt = self._read_front_matter(document_id)

        from api.dependencies import get_llm_client

        llm = get_llm_client()
        enricher = BaseEnricher(llm)
        from utils.prompt_budget import PromptBudget, PromptBudgetError, allocate_document_sample

        domain = use_domain_context(document_id).clause(excerpt)
        fixed_prefix = (
            "You are a library cataloging assistant.\n\n"
            f"{domain}"
            "TASK: Extract bibliographic metadata from the document excerpt below.\n"
            "Return ONLY valid JSON with these keys (use empty string if unknown):\n"
            "{\n"
            '  "title_original": "The full title exactly as printed, in the document own '
            'language",\n'
            '  "title_vi": "The title in Vietnamese, in the standard terminology of the '
            "document's specialty — not a word-by-word translation (e.g. Ballistics → "
            'Đạn đạo học). For a Vietnamese document, the title itself",\n'
            '  "title_en": "The title in English (the original when it is already English)",\n'
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

        self._progress(task_id, 50, "Đang trích xuất metadata")
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
        # Order and brackets are code, not model output: «Gốc (Việt/Anh)».
        if str(meta.get("title_original") or "").strip():
            defaults["title_display"] = format_title_display(
                str(meta.get("title_original")),
                str(meta.get("title_vi") or ""),
                str(meta.get("title_en") or ""),
                source_language,
            )
        if not defaults["title_display"]:
            defaults["title_display"] = doc_title

        self._progress(task_id, 90, "Đang lưu metadata")
        with db_manager.session() as db:
            from data.db_models import Document

            row = db.query(Document).filter(Document.id == document_id).first()
            if row:
                row.bibliographic_metadata = defaults

        self._progress(task_id, 100, "Hoàn tất")
        return defaults
