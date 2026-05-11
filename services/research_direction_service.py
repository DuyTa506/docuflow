"""
Research direction identification service.

Uses LLM to identify research directions from document text,
matching against predefined catalog and suggesting new ones.
"""
from config.settings import lang_name, settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from data.db_models import (
    ResearchDirection,
    DocumentResearchDirection,
    ResearchExtraction,
)
from services.base_service import BaseTaskService
from services.task_manager import task_manager


class ResearchDirectionService(BaseTaskService):
    """Research direction extraction (background task)."""

    def submit(self, db, document_id: str) -> tuple:
        """Create a ResearchExtraction(PENDING) record and submit task.

        Returns (task_id, extraction_id).
        """
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")

        extraction = ResearchExtraction(document_id=document_id, status="PENDING")
        db.add(extraction)
        db.commit()
        db.refresh(extraction)
        extraction_id = extraction.id

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="RESEARCH_DIRECTIONS",
            coro=self._extract(document_id, extraction_id),
        )
        return task_id, extraction_id

    async def _extract(self, document_id: str, extraction_id: str = None):
        db_manager = get_db_manager()

        def _update_extraction(**fields):
            if not extraction_id:
                return
            with db_manager.session() as db:
                e = db.query(ResearchExtraction).filter(ResearchExtraction.id == extraction_id).first()
                if e:
                    for k, v in fields.items():
                        setattr(e, k, v)

        _update_extraction(status="IN_PROGRESS")

        try:
            return await self._do_extract(document_id, extraction_id)
        except Exception as exc:
            _update_extraction(status="FAILED", error=str(exc))
            raise

    async def _do_extract(self, document_id: str, extraction_id: str = None):
        db_manager = get_db_manager()

        text = self._read_text(document_id)

        # Get predefined catalog (needs its own session query — not in base helper)
        with db_manager.session() as db:
            predefined = db.query(ResearchDirection).filter(
                ResearchDirection.is_predefined == True
            ).all()
            catalog_names = [rd.direction_name for rd in predefined]

        task_id = self._find_task_id(document_id, "RESEARCH_DIRECTIONS")

        from api.dependencies import get_llm_client
        llm = get_llm_client()

        self._progress(task_id, 20, "Analyzing research directions")

        catalog_text = "\n".join(f"- {n}" for n in catalog_names) if catalog_names else "(empty catalog)"
        doc_text = BaseEnricher(llm).truncate_to_tokens(text, settings.ai_chunk_tokens - 1000)
        out_lang = lang_name(settings.research_output_lang)

        prompt = (
            "You are a research analysis expert. Analyze how this document relates to "
            "a catalog of known research directions.\n\n"
            f"CATALOG OF KNOWN DIRECTIONS:\n{catalog_text}\n\n"
            "TASK: Identify all research directions discussed in the document.\n"
            "For EACH direction, determine whether it matches a catalog entry or is new.\n\n"
            "MATCHING CRITERIA:\n"
            "- A direction is \"predefined\" if the document discusses substantially the same topic "
            "as a catalog entry (shared key terms, claims, or methodology).\n"
            "- A direction is \"new\" if it is substantively present in the document but absent from the catalog.\n"
            "- If a direction appears both in the catalog AND the document, mark it as predefined.\n\n"
            "REASONING (chain of thought): For each direction, first write your reasoning about "
            "why it matches a catalog entry or why it is new. Then assign a confidence score.\n\n"
            "CONFIDENCE CALIBRATION:\n"
            "- 0.9-1.0: Explicitly named, central topic of the document\n"
            "- 0.7-0.89: Clearly discussed but not the main focus\n"
            "- 0.5-0.69: Tangentially mentioned or implied; include only if material\n"
            "- <0.5: do not include\n\n"
            "Return ONLY valid JSON as a list:\n"
            '[{"direction_name": "...", "is_predefined": true/false, '
            '"confidence": 0.85, "reasoning": "..."}, ...]\n\n'
            f"Write `direction_name` and `reasoning` in {out_lang}. "
            "Catalog names already in their original language stay as-is.\n\n"
            f"DOCUMENT:\n{doc_text}\n\nJSON:"
        )

        response = await llm.chat_completion(prompt)

        self._progress(task_id, 70, "Storing directions")

        directions_list = self._extract_json(llm, response, list_key="directions")

        # Store
        with db_manager.session() as db:
            # Remove existing associations
            db.query(DocumentResearchDirection).filter(
                DocumentResearchDirection.document_id == document_id
            ).delete()

            stored = []
            for item in directions_list:
                dir_name = item.get("direction_name", "").strip()
                if not dir_name:
                    continue
                confidence = float(item.get("confidence", 0.5))
                reasoning = item.get("reasoning", "")
                is_predef = item.get("is_predefined", False)

                # Find or create direction
                rd = db.query(ResearchDirection).filter(
                    ResearchDirection.direction_name == dir_name
                ).first()
                if not rd:
                    rd = ResearchDirection(
                        direction_name=dir_name,
                        is_predefined=is_predef,
                    )
                    db.add(rd)
                    db.flush()

                assoc = DocumentResearchDirection(
                    document_id=document_id,
                    direction_id=rd.id,
                    confidence=confidence,
                    reasoning=reasoning,
                )
                db.add(assoc)
                stored.append({
                    "direction_name": dir_name,
                    "confidence": confidence,
                    "is_predefined": is_predef,
                })

            # Mark extraction COMPLETED in same session
            if extraction_id:
                e = db.query(ResearchExtraction).filter(ResearchExtraction.id == extraction_id).first()
                if e:
                    e.status = "COMPLETED"
                    e.total_directions = len(stored)

        return {"directions": stored, "count": len(stored)}
