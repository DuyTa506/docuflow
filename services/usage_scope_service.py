"""Map a document onto training disciplines / strong research groups (§3).

Drives §3 from the BGD catalog plus the LLM.
"""

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
    catalog_text_block,
    has_entries,
    load_catalog,
    normalize_name,
    resolve_items,
    response_schema,
)
from utils.digest_format import usage_scope_defaults

logger = logging.getLogger(__name__)

# Level labels for the prompt. Catalog and output are both Vietnamese; the JSON
# key travels alongside so the model knows where to put each list.
LEVEL_LABELS = {
    "undergraduate": "ĐẠI HỌC",
    "master": "THẠC SĨ",
    "phd": "TIẾN SĨ",
}

# What each level actually asks. Without these the prompt posed one question and
# printed the answer into three keys: over six live runs on DOC_002 the doctoral
# list was never anything but a subset of the other two.
LEVEL_CRITERIA = {
    "undergraduate": (
        "does this teach foundation material a curriculum builds on? "
        "Textbooks, introductions, standard techniques"
    ),
    "master": (
        "does this deepen or specialise beyond the foundation, for advanced "
        "coursework or a thesis?"
    ),
    "phd": (
        "does this support ORIGINAL research: state of the art, methods, open "
        "problems, a reference a researcher would cite?"
    ),
}

# Strong research groups no longer come from a fixed list, so they have no
# natural bound either. Cap them so a rambling answer cannot swallow all of §3.
MAX_RESEARCH_GROUPS = 8


class UsageScopeService(BaseTaskService):
    """Pick catalog disciplines and strong research groups for §3."""

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

        # The discipline catalog is not always present. With no catalog there is
        # nothing to choose from, so skip the call entirely rather than asking the
        # model to pick from an empty list — and say so out loud, because an empty
        # §3 from "no catalog loaded" is a different fact from an empty §3 from
        # "no discipline fits".
        if not has_entries(catalog):
            logger.warning(
                "No CTĐT catalog loaded (source: %s) — skipping §3 usage scope for %s",
                catalog_source(),
                document_id,
            )
            self._progress(task_id, 100, "Không có danh mục CTĐT — bỏ qua mục §3")
            self._save(db_manager, document_id, result)
            return result

        text = self._read_text(document_id)

        from api.dependencies import get_llm_client

        llm = get_llm_client()
        enricher = BaseEnricher(llm)
        from utils.prompt_budget import PromptBudget, PromptBudgetError, allocate_document_sample, build_pipeline_sample

        # Only offer levels that actually have disciplines. An empty section in
        # the prompt is an invitation to invent something to fill it.
        available = [key for key in CATALOG_KEYS if catalog_text_block(catalog, key)]
        catalog_text = "\n\n".join(
            f"{LEVEL_LABELS[key]} ({key}):\n{catalog_text_block(catalog, key)}" for key in available
        )
        # The level rules are built from the same list, so a level with no
        # disciplines is never named — naming it is the invitation to invent.
        level_rules = "".join(
            f"  * {key} ({LEVEL_LABELS[key]}) — {LEVEL_CRITERIA[key]}\n" for key in available
        )

        fixed_prefix = (
            "You are a research librarian assigning a holding to training programmes.\n\n"
            "TASK: Decide which training disciplines below could USE this document as "
            "teaching or study material, then return their CODES.\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "undergraduate": ["7-digit codes"],\n'
            '  "master": ["7-digit codes"],\n'
            '  "phd": ["7-digit codes"],\n'
            '  "strong_research_groups": ["research directions, free text"]\n'
            "}\n\n"
            "RULES:\n"
            "- For the three level keys, return ONLY 7-digit codes copied from the "
            "catalog below. A 5-digit code is a group heading, not a discipline — "
            "never return one.\n"
            "- Do NOT invent codes. A code that is not in the catalog is discarded.\n"
            "- Ask 'would a student on this discipline benefit from this material?', NOT "
            "'is this discipline the document's subject?'. A foundational textbook usually "
            "serves SEVERAL disciplines across a faculty; list every one it supports.\n"
            "- Include a discipline when the document covers a subject its curriculum "
            "builds on, even if the document never names the discipline.\n"
            "- Only leave a level empty when no discipline at that level relates to the "
            "subject at all.\n"
            "- Decide each level SEPARATELY. They ask different questions:\n"
            f"{level_rules}"
            "- The lists are expected to differ, in membership and in length. "
            "A foundational textbook is widest at undergraduate and narrow at "
            "doctoral; a research monograph runs the other way. Returning the same "
            "list at every level means the levels were not judged.\n"
            "- The catalog is not the same at every level, so a code that exists at "
            "one level may not exist at another. Copy each code from the level "
            "block it appears in.\n"
            f"- For 'strong_research_groups', name up to {MAX_RESEARCH_GROUPS} research "
            "directions this document supports, in your own words — this one is NOT "
            "restricted to the catalog.\n"
            "- The groups must be mutually distinct. Never list both a group and a "
            "narrower version of that same group; pick whichever single one the "
            "document actually supports and spend the remaining slots elsewhere.\n\n"
            f"DANH MỤC NGÀNH ĐÀO TẠO:\n{catalog_text}\n\n"
            f"{pipeline_output_lang_clause(json_values=True)}"
        )
        fixed_suffix = (
            f"{pipeline_output_lang_clause(json_values=True)}"
            "Judge each level separately before answering: "
            + "; ".join(f"{key} = {LEVEL_CRITERIA[key].split('?')[0]}" for key in available)
            + ". Identical lists mean the levels were not judged.\n\n"
            "JSON:"
        )
        budget = PromptBudget(
            context_tokens=settings.ai_model_context_window,
            output_reserve=settings.ai_output_reserve_tokens,
        )
        try:
            excerpt, budget_meta = allocate_document_sample(
                document_id=document_id,
                text=text,
                enricher=enricher,
                budget=budget,
                fixed_parts=[
                    fixed_prefix,
                    "DOCUMENT EXCERPT:\n",
                    fixed_suffix,
                ],
                sample_builder=lambda sample_budget: build_pipeline_sample(
                    document_id, text, enricher, sample_budget
                ),
            )
        except PromptBudgetError as exc:
            logger.error("§3 prompt budget exceeded for %s: %s", document_id, exc)
            raise ValueError(f"Usage scope prompt exceeds context window: {exc}") from exc

        prompt = f"{fixed_prefix}DOCUMENT EXCERPT:\n{excerpt}\n\n{fixed_suffix}"
        logger.info("usage_scope prompt budget document_id=%s meta=%s", document_id, budget_meta)

        self._progress(task_id, 40, "Đang xác định phạm vi ứng dụng")
        response = await self._complete(llm, prompt, catalog)

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

        for key in CATALOG_KEYS:
            items = scope.get(key) or []
            if not isinstance(items, list):
                continue
            kept, dropped = resolve_items(catalog, key, items)
            result[key] = kept
            if dropped:
                # These used to vanish without trace, which is how a code with
                # one wrong digit read as "no discipline fits".
                logger.warning(
                    "usage_scope %s: dropped %d item(s) absent from the catalog (%s): %s",
                    document_id,
                    len(dropped),
                    key,
                    "; ".join(str(d) for d in dropped[:10]),
                )

        result["strong_research_groups"] = _clean_research_groups(
            scope.get("strong_research_groups")
        )

        self._progress(task_id, 90, "Đang lưu phạm vi ứng dụng")
        self._save(db_manager, document_id, result)

        self._progress(task_id, 100, "Hoàn tất")
        return result

    @staticmethod
    async def _complete(llm, prompt: str, catalog: dict) -> str:
        """Ask the model, constraining the answer to codes that exist.

        Only the provider can enforce this, and not every OpenAI-compatible
        server does — some ignore `response_format`, which leaves us exactly
        where we were, and some reject the request outright. Losing §3 over an
        unsupported parameter is worse than one extra call, so a failure retries
        once without it; `resolve_items` still guards the unconstrained answer.
        A second failure is a real outage and propagates.
        """
        schema = response_schema(catalog)
        try:
            return await llm.chat_completion(
                prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "usage_scope", "schema": schema},
                },
            )
        except Exception as exc:
            logger.warning(
                "Provider rejected the §3 response schema (%s) — retrying unconstrained", exc
            )
            return await llm.chat_completion(prompt)

    def _save(self, db_manager, document_id: str, result: dict) -> None:
        with db_manager.session() as db:
            from data.db_models import Document

            row = db.query(Document).filter(Document.id == document_id).first()
            if row:
                row.usage_scope = result


# Thresholds for folding a narrower branch into a broader group already on the
# list. Both numbers lean towards **keeping**: dropping a real group loses
# information from an official document, while keeping a redundant pair only
# costs one of eight slots.
#
# Vietnamese splits compounds into separate syllables, so syllable prefixes
# collide easily: "Kỹ thuật điện" is a prefix of "Kỹ thuật điện tử" yet the two
# disciplines are entirely different. Requiring a qualifier of >= 3 syllables
# ("hiệu năng cao") rules those out. The real semantic judgement is pushed to the
# prompt, the only place that can make it.
_MIN_BASE_TOKENS = 3
_MIN_EXTRA_TOKENS = 3


def _is_specialisation(tokens: list[str], base: list[str]) -> bool:
    """`tokens` is `base` plus a qualifier long enough to count."""
    return (
        len(base) >= _MIN_BASE_TOKENS
        and len(tokens) - len(base) >= _MIN_EXTRA_TOKENS
        and tokens[: len(base)] == base
    )


def _clean_research_groups(items) -> list[str]:
    """Clean the strong-research-group list — the one field with no catalog.

    No catalog to check against means no guard against invention; only the
    mechanical part is possible here: drop empty and wrongly-typed entries, trim
    whitespace, de-duplicate on the normalised name, fold narrow branches into
    broader groups, and cap the count.

    De-duplicating on the normalised name does not catch "Kiến trúc máy tính"
    sitting next to "Kiến trúc máy tính hiệu năng cao" — two different strings,
    and N4.11.160 returned exactly that pair. The earlier entry is the one kept:
    the model orders by descending relevance.
    """
    if not isinstance(items, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    kept_tokens: list[list[str]] = []
    for item in items:
        if not isinstance(item, str):
            continue
        name = " ".join(item.split())
        key = normalize_name(name)
        if not key or key in seen:
            continue
        tokens = key.split()
        if any(
            _is_specialisation(tokens, prev) or _is_specialisation(prev, tokens)
            for prev in kept_tokens
        ):
            continue
        seen.add(key)
        kept_tokens.append(tokens)
        cleaned.append(name)
        if len(cleaned) >= MAX_RESEARCH_GROUPS:
            break
    return cleaned
