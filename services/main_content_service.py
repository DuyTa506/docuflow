"""
Main content extraction service.

Extracts structured key points, methods, results, and conclusions
from document text using LLM.
"""
from config.settings import settings
from core.pageindex.enrichment.base import BaseEnricher
from data.database import get_db_manager
from data.db_models import MainContent
from services.base_service import BaseTaskService
from services.task_manager import task_manager


class MainContentService(BaseTaskService):
    """Extract structured main content (background task)."""

    def submit(self, db, document_id: str) -> tuple:
        """Create a MainContent(PENDING) record and submit background task.

        Returns (task_id, main_content_id).
        """
        from data.repositories import DocumentRepository
        repo = DocumentRepository(db)
        if not repo.get(document_id):
            raise ValueError("Document not found")

        mc = MainContent(document_id=document_id, status="PENDING")
        db.add(mc)
        db.commit()
        db.refresh(mc)
        main_content_id = mc.id

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="MAIN_CONTENT",
            coro=self._extract(document_id, main_content_id),
        )
        return task_id, main_content_id

    async def _extract(self, document_id: str, main_content_id: str = None):
        db_manager = get_db_manager()

        def _set_status(status: str):
            if not main_content_id:
                return
            with db_manager.session() as db:
                mc = db.query(MainContent).filter(MainContent.id == main_content_id).first()
                if mc:
                    mc.status = status

        _set_status("IN_PROGRESS")

        try:
            text = self._read_text(document_id)
            task_id = self._find_task_id(document_id, "MAIN_CONTENT")

            from api.dependencies import get_llm_client
            llm = get_llm_client()

            self._progress(task_id, 20, "Analyzing document")

            doc_text = BaseEnricher(llm).truncate_to_tokens(text, settings.ai_chunk_tokens - 1000)
            prompt = (
                "You are a scientific document analyst. Extract structured information "
                "from the document below.\n\n"
                "TASK: Return ONLY valid JSON with these keys:\n"
                '{\n'
                '  "key_points": ["list of 3-7 main claims or contributions — each a complete sentence"],\n'
                '  "methods": ["list of specific methods, algorithms, or approaches — include named variants"],\n'
                '  "results": ["list of key results or findings — include numbers, metrics, and comparisons where present"],\n'
                '  "conclusions": ["list of conclusions the authors draw — distinct from results"]\n'
                '}\n\n'
                "EXTRACTION RULES:\n"
                "- key_points: Extract only claims the document explicitly makes. Do NOT infer.\n"
                "- methods: Include named methods, tools, and dataset names. Each entry must be concrete.\n"
                "- results: If a number or statistic is mentioned, include it verbatim. "
                "Example: \"accuracy improved from 82% to 91%\".\n"
                "- conclusions: These are INTERPRETATIONS of results — distinguish them from raw observations. "
                "If the document separates Discussion from Results, use that distinction.\n\n"
                "GROUNDING: Each entry must be directly stated in the source text. "
                "If uncertain about a claim, omit it. Do NOT fabricate data or references.\n\n"
                f"DOCUMENT:\n{doc_text}\n\nJSON:"
            )

            response = await llm.chat_completion(prompt)

            self._progress(task_id, 80, "Parsing results")

            try:
                details = llm.extract_json(response)
            except Exception:
                details = {
                    "key_points": [response.strip()],
                    "methods": [],
                    "results": [],
                    "conclusions": [],
                }

            # Update existing record
            with db_manager.session() as db:
                if main_content_id:
                    mc = db.query(MainContent).filter(MainContent.id == main_content_id).first()
                    if mc:
                        mc.details = details
                        mc.status = "COMPLETED"
                else:
                    db.add(MainContent(
                        document_id=document_id,
                        details=details,
                        status="COMPLETED",
                    ))

            return details
        except Exception:
            _set_status("FAILED")
            raise
