"""
Research direction identification service.

Uses LLM to identify research directions from document text,
matching against predefined catalog and suggesting new ones.
"""
import json
from data.database import get_db_manager
from data.db_models import (
    ResearchDirection,
    DocumentResearchDirection,
)
from services.base_service import BaseTaskService
from services.task_manager import task_manager, TaskManager


class ResearchDirectionService(BaseTaskService):
    """Research direction extraction (background task)."""

    def submit(self, db, document_id: str) -> str:
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")
        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="RESEARCH_DIRECTIONS",
            coro=self._extract(document_id),
        )
        return task_id

    async def _extract(self, document_id: str):
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

        prompt = (
            "Analyze the following document and identify its research directions.\n\n"
            f"Existing catalog of known research directions:\n{catalog_text}\n\n"
            "For each direction identified:\n"
            "1. Check if it matches a direction from the catalog\n"
            "2. If not, suggest it as a new direction\n"
            "3. Provide a confidence score (0.0-1.0) and brief reasoning\n\n"
            "Return ONLY valid JSON as a list:\n"
            '[{"direction_name": "...", "is_predefined": true/false, '
            '"confidence": 0.85, "reasoning": "..."}, ...]\n\n'
            f"Document:\n{text[:15000]}\n\nJSON:"
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

        return {"directions": stored, "count": len(stored)}
