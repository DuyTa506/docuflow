"""
Main content extraction service.

Extracts structured key points, methods, results, and conclusions
from document text using LLM.
"""
from data.database import get_db_manager
from data.db_models import MainContent
from services.base_service import BaseTaskService
from services.task_manager import task_manager


class MainContentService(BaseTaskService):
    """Extract structured main content (background task)."""

    def submit(self, db, document_id: str) -> str:
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")
        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="MAIN_CONTENT",
            coro=self._extract(document_id),
        )
        return task_id

    async def _extract(self, document_id: str):
        db_manager = get_db_manager()

        text = self._read_text(document_id)
        task_id = self._find_task_id(document_id, "MAIN_CONTENT")

        from api.dependencies import get_llm_client
        llm = get_llm_client()

        self._progress(task_id, 20, "Analyzing document")

        prompt = (
            "Analyze the following document and extract structured content as JSON.\n"
            "Return ONLY valid JSON with these keys:\n"
            '{\n'
            '  "key_points": ["list of main points"],\n'
            '  "methods": ["list of methods/approaches mentioned"],\n'
            '  "results": ["list of key results/findings"],\n'
            '  "conclusions": ["list of conclusions"]\n'
            '}\n\n'
            f"Document:\n{text[:15000]}\n\nJSON:"
        )

        response = await llm.chat_completion(prompt)

        self._progress(task_id, 80, "Parsing results")

        # Parse JSON from response
        try:
            details = llm.extract_json(response)
        except Exception:
            details = {
                "key_points": [response.strip()],
                "methods": [],
                "results": [],
                "conclusions": [],
            }

        # Store
        with db_manager.session() as db:
            mc = MainContent(
                document_id=document_id,
                details=details,
            )
            db.add(mc)

        return details
