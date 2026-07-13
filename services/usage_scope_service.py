"""Map document to CTĐT / NNC usage scope (§3) using internal catalog + LLM."""

import json
import logging
from pathlib import Path
from typing import Optional

from config.settings import pipeline_output_lang_clause, settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from services.base_service import BaseTaskService
from services.task_manager import task_manager
from utils.digest_format import usage_scope_defaults

logger = logging.getLogger(__name__)


def _load_catalog() -> dict:
    path = Path(__file__).resolve().parent.parent / "config" / "ctdt_catalog.json"
    if not path.is_file():
        return usage_scope_defaults()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class UsageScopeService(BaseTaskService):
    """Select CTĐT and strong research groups from catalog."""

    def submit(self, db, document_id: str) -> tuple[str, bool]:
        from data.repositories import DocumentRepository

        if not DocumentRepository(db).get(document_id):
            raise ValueError("Document not found")

        existing = task_manager.get_active_task_id(db, document_id, "USAGE_SCOPE")
        if existing:
            return existing, True

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="USAGE_SCOPE",
            coro_factory=lambda tid: self._extract(document_id, tid),
        )
        return task_id, False

    async def run_for_pipeline(self, document_id: str, task_id: Optional[str] = None):
        return await self._extract(document_id, task_id)

    async def _extract(self, document_id: str, task_id: Optional[str] = None):
        db_manager = get_db_manager()
        catalog = _load_catalog()
        text = self._read_text(document_id)

        from api.dependencies import get_llm_client

        llm = get_llm_client()
        from utils.doc_sampling import build_pipeline_doc_sample

        excerpt = build_pipeline_doc_sample(
            document_id, text, BaseEnricher(llm), settings.ai_input_budget_tokens
        )

        catalog_text = json.dumps(catalog, ensure_ascii=False, indent=2)
        prompt = (
            "You are an academic curriculum mapping assistant.\n\n"
            "TASK: From the document excerpt, select applicable items ONLY from the catalog below.\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "undergraduate": ["exact strings from catalog undergraduate"],\n'
            '  "master": ["exact strings from catalog master"],\n'
            '  "phd": ["exact strings from catalog phd"],\n'
            '  "strong_research_groups": ["exact strings from catalog strong_research_groups"]\n'
            "}\n\n"
            "RULES:\n"
            "- Each selected string MUST match a catalog entry verbatim.\n"
            "- Do NOT invent new program or group names.\n"
            "- undergraduate/master/phd: training programs. strong_research_groups: inferred "
            "research directions (what HVKTQS researches in that field), NOT program names.\n"
            "- Select only entries clearly relevant to the document topic.\n\n"
            f"CATALOG:\n{catalog_text}\n\n"
            f"{pipeline_output_lang_clause(json_values=True)}"
            f"DOCUMENT EXCERPT:\n{excerpt}\n\n"
            f"{pipeline_output_lang_clause(json_values=True)}"
            "JSON:"
        )

        self._progress(task_id, 40, "Mapping usage scope")
        response = await llm.chat_completion(prompt)

        try:
            scope = llm.extract_json(response)
            if not isinstance(scope, dict):
                scope = {}
        except Exception as exc:
            logger.warning(
                "usage_scope extract_json failed for document %s: %s | response snippet: %r",
                document_id,
                exc,
                response[:200],
            )
            scope = {}

        result = usage_scope_defaults()
        for key in result:
            items = scope.get(key) or []
            if isinstance(items, list):
                allowed = set(catalog.get(key, []))
                result[key] = [s for s in items if s in allowed]

        self._progress(task_id, 90, "Saving usage scope")
        with db_manager.session() as db:
            from data.db_models import Document

            row = db.query(Document).filter(Document.id == document_id).first()
            if row:
                row.usage_scope = result

        self._progress(task_id, 100, "Done")
        return result
