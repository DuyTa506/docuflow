"""Map document to CTĐT / NNC usage scope (§3) using internal catalog + LLM."""

import json
import logging
from typing import Optional

from config.settings import pipeline_output_lang_clause, settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from services.base_service import BaseTaskService
from services.task_manager import task_manager
from utils.ctdt_catalog import (
    CATALOG_KEYS,
    catalog_source,
    has_entries,
    load_catalog,
    resolve_items,
)
from utils.digest_format import usage_scope_defaults

logger = logging.getLogger(__name__)


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
        catalog = load_catalog()
        result = usage_scope_defaults()

        # The Academy's programme list is not always available. Without it there
        # is nothing to select from, so skip the call rather than ask a model to
        # pick from an empty list — and say so, because an empty §3 for "no
        # catalog" and one for "no relevant programme" mean different things.
        if not has_entries(catalog):
            logger.warning(
                "Chưa nạp danh mục CTĐT (nguồn: %s) — bỏ qua §3 phạm vi sử dụng cho %s",
                catalog_source(),
                document_id,
            )
            self._progress(task_id, 100, "Chưa có danh mục CTĐT — bỏ qua §3")
            self._save(db_manager, document_id, result)
            return result

        text = self._read_text(document_id)

        from api.dependencies import get_llm_client

        llm = get_llm_client()
        from utils.doc_sampling import build_pipeline_doc_sample

        excerpt = build_pipeline_doc_sample(
            document_id, text, BaseEnricher(llm), settings.ai_input_budget_tokens
        )

        catalog_text = json.dumps(
            {k: catalog.get(k) or [] for k in CATALOG_KEYS}, ensure_ascii=False, indent=2
        )
        # The question a librarian is answering is "who can USE this book", not
        # "what is this book about". Framed as aboutness — plus a nudge that an
        # empty list is fine — the model returned one programme for a computer
        # architecture textbook that plainly serves every computing programme.
        prompt = (
            "You are a research librarian assigning a holding to training programmes.\n\n"
            "TASK: Decide which of the Academy's programmes below could USE this document "
            "as teaching or study material, then return them.\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "undergraduate": ["exact strings from catalog undergraduate"],\n'
            '  "master": ["exact strings from catalog master"],\n'
            '  "phd": ["exact strings from catalog phd"],\n'
            '  "strong_research_groups": ["exact strings from catalog strong_research_groups"]\n'
            "}\n\n"
            "RULES:\n"
            "- Each selected string MUST match a catalog entry verbatim.\n"
            "- Do NOT invent new program or group names — choose only from the catalog.\n"
            "- Ask 'would a student on this programme benefit from this material?', NOT "
            "'is this programme the document's subject?'. A foundational textbook usually "
            "serves SEVERAL programmes across a faculty; list every one it supports.\n"
            "- Include a programme when the document covers a subject its curriculum "
            "builds on, even if the document never names the programme.\n"
            "- Only leave a level empty when no programme at that level relates to the "
            "subject at all.\n\n"
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

        for key in result:
            items = scope.get(key) or []
            if not isinstance(items, list):
                continue
            kept, dropped = resolve_items(catalog, key, items)
            result[key] = kept
            if dropped:
                # Previously these disappeared without a trace, which is how a
                # near-miss spelling read as "no matching programme".
                logger.warning(
                    "usage_scope %s: bỏ %d mục không có trong danh mục CTĐT (%s): %s",
                    document_id,
                    len(dropped),
                    key,
                    "; ".join(str(d) for d in dropped[:10]),
                )

        self._progress(task_id, 90, "Saving usage scope")
        self._save(db_manager, document_id, result)

        self._progress(task_id, 100, "Done")
        return result

    def _save(self, db_manager, document_id: str, result: dict) -> None:
        with db_manager.session() as db:
            from data.db_models import Document

            row = db.query(Document).filter(Document.id == document_id).first()
            if row:
                row.usage_scope = result
