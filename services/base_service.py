"""
BaseTaskService — shared helpers for all async background-task services.

Eliminates the following copy-pasted patterns across 6+ services:
  * _find_task_id  — "query latest task for (document, type)" (7 copies)
  * _read_text     — "load DigitizedText, raise if missing" (5 copies)
  * _progress      — "open session; call update_progress" (10+ copies)
  * _extract_json  — "llm.extract_json + fallback to []" (4 copies)
"""
import asyncio
from typing import Optional

from data.database import get_db_manager
from services.task_manager import TaskManager


class BaseTaskService:
    """
    Mixin for all async document-processing services.

    Subclasses inherit helpers below; no constructor required.
    """

    # ── Task ID lookup ───────────────────────────────────────────────

    def _find_task_id(self, document_id: str, task_type: str) -> Optional[str]:
        """
        Return the ID of the most recently created task of *task_type* for
        *document_id*, or None if none exists.
        """
        db_manager = get_db_manager()
        with db_manager.session() as db:
            from data.repositories import TaskRepository
            repo = TaskRepository(db)
            task = repo.find_latest(document_id, task_type)
            return task.id if task else None

    # ── Text loading ─────────────────────────────────────────────────

    def _has_digitized_text(self, document_id: str) -> bool:
        """Return True when normalized or OCR text is available."""
        db_manager = get_db_manager()
        with db_manager.session() as db:
            from data.repositories import DocumentRepository
            repo = DocumentRepository(db)
            dt = repo.get_digitized_text(document_id)
            if not dt:
                return False
            return bool((dt.normalized_content or dt.ocr_content or "").strip())

    async def _wait_for_digitized_text(
        self,
        document_id: str,
        *,
        task_id: Optional[str] = None,
        timeout_seconds: int = 7200,
        poll_seconds: float = 5.0,
    ) -> None:
        """Block until OCR/extraction text exists or extraction has failed/timed out."""
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            if self._has_digitized_text(document_id):
                return

            db_manager = get_db_manager()
            with db_manager.session() as db:
                from data.db_models import Document
                from data.repositories import TaskRepository

                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc and doc.processing_status == "FAILED":
                    raise ValueError("Document extraction failed")

                extract_task = TaskRepository(db).find_latest(document_id, "EXTRACT")
                if extract_task and extract_task.status == "FAILED":
                    err = extract_task.error or "extraction failed"
                    raise ValueError(f"Document extraction failed: {err}")

                if doc and doc.processing_status == "EXTRACTED":
                    # Status says done but text row missing — keep polling briefly.
                    pass
                elif not extract_task or extract_task.status not in (
                    "PENDING",
                    "RUNNING",
                    "COMPLETED",
                ):
                    break

            self._progress(
                task_id,
                5,
                "Waiting for OCR/extraction to finish…",
            )
            await asyncio.sleep(poll_seconds)

        if not self._has_digitized_text(document_id):
            raise ValueError("No digitized text — run OCR/extraction first")

    def _read_text(self, document_id: str) -> str:
        """
        Load normalized or OCR text for *document_id*.

        Raises:
            ValueError: if no DigitizedText row exists, or the row has no text.
        """
        db_manager = get_db_manager()
        with db_manager.session() as db:
            from data.repositories import DocumentRepository
            repo = DocumentRepository(db)
            dt = repo.get_digitized_text(document_id)
            if not dt:
                raise ValueError("No digitized text — run OCR/extraction first")
            text = dt.normalized_content or dt.ocr_content
            if not text:
                raise ValueError("No text content available")
            return text

    # ── Progress reporting ───────────────────────────────────────────

    def _progress(self, task_id: Optional[str], pct: int, msg: str) -> None:
        """
        Update task progress in the DB.  Safe to call with *task_id=None*
        (no-op), so callers never need to guard against missing task IDs.
        """
        if not task_id:
            return
        db_manager = get_db_manager()
        with db_manager.session() as db:
            TaskManager.update_progress(db, task_id, pct, msg)

    # ── JSON extraction ──────────────────────────────────────────────

    def _extract_json(self, llm, response: str, list_key: str = None) -> list:
        """
        Parse a JSON list from an LLM response string.

        Tries ``llm.extract_json()``, then checks if the result is already a
        list, or if *list_key* exists in the parsed dict.  Returns ``[]`` on
        any failure — callers should handle the empty-list case gracefully.

        Args:
            llm:       LLM client (must have ``extract_json`` method).
            response:  Raw LLM completion string.
            list_key:  Dict key to unwrap if the JSON root is an object.

        Returns:
            Parsed list or [].
        """
        try:
            raw = llm.extract_json(response)
            if isinstance(raw, list):
                return raw
            if isinstance(raw, dict) and list_key and list_key in raw:
                return raw[list_key]
            return []
        except Exception:
            return []
