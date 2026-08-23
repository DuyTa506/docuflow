"""
Research direction identification service.

Uses LLM to identify research directions from document text,
matching against predefined catalog and suggesting new ones.
"""

import logging
from typing import Optional

from config.settings import pipeline_output_lang_clause, settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from data.db_models import (
    DocumentResearchDirection,
    ResearchDirection,
    ResearchExtraction,
)
from services.base_service import BaseTaskService
from services.task_manager import task_manager
from utils.ctdt_catalog import name_key

logger = logging.getLogger(__name__)


class ResearchDirectionService(BaseTaskService):
    """Research direction extraction (background task)."""

    def submit(self, db, document_id: str) -> tuple:
        """Create a ResearchExtraction(PENDING) record and submit task.

        Returns (task_id, extraction_id, reused).
        """
        from data.repositories import DocumentRepository

        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")

        existing_task = task_manager.get_active_task_id(db, document_id, "RESEARCH_DIRECTIONS")
        if existing_task:
            extraction = (
                db.query(ResearchExtraction)
                .filter(ResearchExtraction.document_id == document_id)
                .order_by(ResearchExtraction.created_at.desc())
                .first()
            )
            return existing_task, (extraction.id if extraction else None), True

        extraction = ResearchExtraction(document_id=document_id, status="PENDING")
        db.add(extraction)
        db.commit()
        db.refresh(extraction)
        extraction_id = extraction.id

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="RESEARCH_DIRECTIONS",
            coro_factory=lambda tid: self._extract(document_id, extraction_id, tid),
        )
        return task_id, extraction_id, False

    async def submit_async(self, db, document_id: str) -> tuple:
        """Temporal-aware submit — see SummarizationService.submit_async."""
        from config.settings import settings

        if not settings.stage_rerun_use_temporal:
            return self.submit(db, document_id)

        from services.stage_dispatch import submit_stage_with_resource

        return await submit_stage_with_resource(
            db, document_id, "RESEARCH_DIRECTIONS", ResearchExtraction
        )

    async def run_for_pipeline(self, document_id: str, task_id: Optional[str] = None):
        db_manager = get_db_manager()
        with db_manager.session() as db:
            e = (
                db.query(ResearchExtraction)
                .filter(ResearchExtraction.document_id == document_id)
                .order_by(ResearchExtraction.created_at.desc())
                .first()
            )
            if not e:
                e = ResearchExtraction(document_id=document_id, status="PENDING")
                db.add(e)
                db.commit()
                db.refresh(e)
            extraction_id = e.id
        return await self._do_extract(document_id, extraction_id, task_id)

    # ── Extraction bookkeeping ───────────────────────────────────────

    def _update_extraction(self, extraction_id: Optional[str], **fields) -> None:
        if not extraction_id:
            return
        db_manager = get_db_manager()
        with db_manager.session() as db:
            e = db.query(ResearchExtraction).filter(ResearchExtraction.id == extraction_id).first()
            if e:
                for k, v in fields.items():
                    setattr(e, k, v)

    async def _extract(
        self,
        document_id: str,
        extraction_id: str = None,
        task_id: str = None,
    ):
        self._update_extraction(extraction_id, status="IN_PROGRESS")

        try:
            return await self._do_extract(document_id, extraction_id, task_id)
        except Exception as exc:
            self._update_extraction(extraction_id, status="FAILED", error=str(exc))
            raise

    async def _do_extract(
        self,
        document_id: str,
        extraction_id: str = None,
        task_id: str = None,
    ):
        db_manager = get_db_manager()

        text = self._read_text(document_id)

        # Get predefined catalog (needs its own session query — not in base helper)
        with db_manager.session() as db:
            predefined = (
                db.query(ResearchDirection).filter(ResearchDirection.is_predefined == True).all()
            )
            catalog_names = [rd.direction_name for rd in predefined]

        from api.dependencies import get_llm_client

        llm = get_llm_client()

        self._progress(task_id, 20, "Đang phân tích hướng nghiên cứu")

        enricher = BaseEnricher(llm)
        from utils.prompt_budget import PromptBudget, PromptBudgetError, allocate_document_sample, build_pipeline_sample

        lang_clause = pipeline_output_lang_clause(json_values=True)
        max_items = int(settings.research_directions_max_items)

        catalog_block = (
            "The Academy already works in these areas (context only — you may and should "
            "propose directions beyond this list):\n"
            + "\n".join(f"- {n}" for n in catalog_names)
            + "\n\n"
            if catalog_names
            else ""
        )

        fixed_prefix = (
            "You are a senior research advisor. Read the document, then propose the "
            "research directions it opens up.\n\n"
            f"{catalog_block}"
            f"TASK: Propose up to {max_items} research directions connected to this document.\n"
            "Draw on your own knowledge of the field — a useful direction is often one the "
            "document does not name, but which its content makes worth pursuing: an open "
            "problem it exposes, a method it could be extended with, an application it "
            "enables, or a gap between what it covers and where the field has moved.\n\n"
            "CONSTRAINTS:\n"
            "- Each direction must be traceable to something in the document — a topic, "
            "method, limitation or claim. Say which, in `reasoning`.\n"
            "- Prefer specific, actionable directions over broad field names.\n"
            "- Do not repeat the same direction under different wording.\n\n"
            "CONFIDENCE CALIBRATION (how strongly the document supports the direction):\n"
            "- 0.9-1.0: The document works directly on this; an obvious next step\n"
            "- 0.7-0.89: Solidly implied by the document's content or limitations\n"
            "- 0.5-0.69: A plausible extension requiring knowledge beyond the document\n"
            "- <0.5: do not include\n\n"
            "Return ONLY valid JSON as a list:\n"
            '[{"direction_name": "...", "confidence": 0.85, "reasoning": "..."}, ...]\n'
            "Keep `reasoning` to one short sentence — the whole answer must fit in the "
            "response budget, and a truncated JSON list is unusable.\n\n"
            f"{lang_clause}"
            "Write `direction_name` and `reasoning` in Vietnamese.\n\n"
            "DOCUMENT:\n"
        )
        fixed_suffix = f"\n\n{lang_clause}JSON:"

        budget = PromptBudget(
            context_tokens=settings.ai_model_context_window,
            output_reserve=settings.research_directions_max_tokens,
        )
        try:
            doc_text, budget_meta = allocate_document_sample(
                document_id=document_id,
                text=text,
                enricher=enricher,
                budget=budget,
                fixed_parts=[fixed_prefix, fixed_suffix],
                sample_builder=lambda sample_budget: build_pipeline_sample(
                    document_id, text, enricher, sample_budget
                ),
            )
        except PromptBudgetError as exc:
            logger.error("Research directions prompt budget exceeded for %s: %s", document_id, exc)
            raise ValueError(f"Research directions prompt exceeds context window: {exc}") from exc

        prompt = f"{fixed_prefix}{doc_text}{fixed_suffix}"
        logger.info(
            "research_directions prompt budget document_id=%s meta=%s",
            document_id,
            budget_meta,
        )

        response = await llm.chat_completion(
            prompt, max_tokens=settings.research_directions_max_tokens
        )

        self._progress(task_id, 70, "Đang lưu hướng nghiên cứu")

        directions_list, parse_failed = self._parse_json_list(llm, response, list_key="directions")

        # A truncated or malformed answer is a stage failure, not an empty
        # result: previously it silently deleted the previous run's directions
        # and reported COMPLETED with zero rows.
        if parse_failed:
            msg = (
                "research_directions: phản hồi LLM không phải JSON hợp lệ "
                "(có thể bị cắt cụt) — giữ nguyên kết quả lần chạy trước"
            )
            logger.warning("%s | document=%s", msg, document_id)
            self._update_extraction(extraction_id, status="FAILED", error=msg)
            raise ValueError(msg)

        if not directions_list:
            # Genuinely nothing found. Keep whatever a previous run stored.
            with db_manager.session() as db:
                kept = (
                    db.query(DocumentResearchDirection)
                    .filter(DocumentResearchDirection.document_id == document_id)
                    .count()
                )
                self._mark_completed(db, extraction_id, kept)
            logger.warning(
                "research_directions: không tìm được hướng nghiên cứu nào cho %s "
                "(giữ lại %s liên kết cũ)",
                document_id,
                kept,
            )
            return {"directions": [], "count": 0}

        # Catalog membership is a lookup on our side, not a claim by the model:
        # it used to answer `is_predefined` itself and got it both ways round.
        catalog_lookup = {name_key(n): n for n in catalog_names}

        # Store
        with db_manager.session() as db:
            # Remove existing associations
            db.query(DocumentResearchDirection).filter(
                DocumentResearchDirection.document_id == document_id
            ).delete()

            # `direction_name` is UNIQUE and used to be looked up by exact
            # string, so every rephrasing became a permanent new row — the
            # table reached 290 entries of near-copies. Match order-insensitively
            # against what is already there instead.
            existing_by_key = {}
            for row in db.query(ResearchDirection).all():
                existing_by_key.setdefault(name_key(row.direction_name), row)

            stored = []
            seen_keys: set[tuple[str, ...]] = set()
            for item in directions_list[:max_items]:
                if not isinstance(item, dict):
                    continue
                dir_name = str(item.get("direction_name", "")).strip()
                if not dir_name:
                    continue
                try:
                    confidence = float(item.get("confidence", 0.5))
                except (TypeError, ValueError):
                    confidence = 0.5
                reasoning = item.get("reasoning", "")

                key = name_key(dir_name)
                canonical = catalog_lookup.get(key)
                is_predef = canonical is not None
                if canonical:
                    dir_name = canonical
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                # Find or create direction
                rd = existing_by_key.get(key)
                if rd is not None:
                    dir_name = rd.direction_name
                else:
                    rd = ResearchDirection(
                        direction_name=dir_name,
                        is_predefined=is_predef,
                    )
                    db.add(rd)
                    db.flush()
                    existing_by_key[key] = rd

                assoc = DocumentResearchDirection(
                    document_id=document_id,
                    direction_id=rd.id,
                    confidence=confidence,
                    reasoning=reasoning,
                )
                db.add(assoc)
                stored.append(
                    {
                        "direction_name": dir_name,
                        "confidence": confidence,
                        "is_predefined": is_predef,
                    }
                )

            self._mark_completed(db, extraction_id, len(stored))

        return {"directions": stored, "count": len(stored)}

    @staticmethod
    def _mark_completed(db, extraction_id: Optional[str], total: int) -> None:
        if not extraction_id:
            return
        e = db.query(ResearchExtraction).filter(ResearchExtraction.id == extraction_id).first()
        if e:
            e.status = "COMPLETED"
            e.total_directions = total
