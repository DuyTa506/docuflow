"""
Keyword extraction service.

Uses LLM to extract keywords with importance scores from document text.
"""
import json
from data.database import get_db_manager
from data.db_models import Keyword, DocumentKeyword
from services.base_service import BaseTaskService
from services.task_manager import task_manager, TaskManager


class KeywordService(BaseTaskService):
    """Keyword extraction (background task)."""

    def submit(self, db, document_id: str, max_keywords: int = 20) -> str:
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")
        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="KEYWORDS",
            coro=self._extract(document_id, max_keywords),
        )
        return task_id

    async def _extract(self, document_id: str, max_keywords: int):
        db_manager = get_db_manager()

        text = self._read_text(document_id)
        task_id = self._find_task_id(document_id, "KEYWORDS")

        from api.dependencies import get_llm_client
        llm = get_llm_client()

        self._progress(task_id, 20, "Extracting keywords")

        prompt = (
            f"Extract the top {max_keywords} keywords or key phrases from the following "
            "document. For each keyword, provide an importance weight from 0.0 to 1.0.\n"
            "Return ONLY valid JSON as a list of objects:\n"
            '[{"keyword": "example term", "weight": 0.95}, ...]\n\n'
            f"Document:\n{text[:15000]}\n\nJSON:"
        )

        response = await llm.chat_completion(prompt)

        self._progress(task_id, 60, "Parsing keywords")

        keywords_list = self._extract_json(llm, response, list_key="keywords")

        # Store keywords
        with db_manager.session() as db:
            # Remove existing associations for this document
            db.query(DocumentKeyword).filter(DocumentKeyword.document_id == document_id).delete()

            stored = []
            for item in keywords_list[:max_keywords]:
                kw_name = item.get("keyword", "").strip()
                weight = float(item.get("weight", 1.0))
                if not kw_name:
                    continue

                # Find or create keyword
                kw = db.query(Keyword).filter(Keyword.keyword_name == kw_name).first()
                if not kw:
                    kw = Keyword(keyword_name=kw_name)
                    db.add(kw)
                    db.flush()

                # Create association
                assoc = DocumentKeyword(
                    document_id=document_id,
                    keyword_id=kw.id,
                    weight=weight,
                )
                db.add(assoc)
                stored.append({"keyword": kw_name, "weight": weight})

        return {"keywords": stored, "count": len(stored)}
