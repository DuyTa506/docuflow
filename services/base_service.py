"""
BaseTaskService — shared helpers for all async background-task services.

Eliminates the following copy-pasted patterns across 6+ services:
  * _read_text     — "load DigitizedText, raise if missing" (5 copies)
  * _progress      — "open session; call update_progress" (10+ copies)
  * _extract_json  — "llm.extract_json + fallback to []" (4 copies)
"""

import asyncio
import logging
from typing import Optional

from data.database import get_db_manager
from services.progress_reporting import emit_progress
from utils.content_storage import get_object_storage, read_text_field

logger = logging.getLogger(__name__)


class BaseTaskService:
    """
    Mixin for all async document-processing services.

    Subclasses inherit helpers below; no constructor required.
    """

    # ── Text loading ─────────────────────────────────────────────────

    def _has_digitized_text(self, document_id: str) -> bool:
        """Return True when normalized or OCR text is available (DB inline or MinIO)."""
        db_manager = get_db_manager()
        with db_manager.session() as db:
            from data.repositories import DocumentRepository

            repo = DocumentRepository(db)
            dt = repo.get_digitized_text(document_id)
            if not dt:
                return False
            if (dt.normalized_content or dt.ocr_content or "").strip():
                return True
            # Large texts are offloaded to MinIO — inline columns stay empty.
            if dt.normalized_content_key or dt.ocr_content_key:
                storage = get_object_storage()
                if dt.normalized_content_key and storage.exists(dt.normalized_content_key):
                    return True
                if dt.ocr_content_key and storage.exists(dt.ocr_content_key):
                    return True
            return False

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
                pipeline=self._pipeline_for_task(task_id),
                phase="waiting_upstream",
                attempt=1,
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
            text = read_text_field(
                inline=dt.normalized_content,
                key=dt.normalized_content_key,
            )
            if not text:
                text = read_text_field(
                    inline=dt.ocr_content,
                    key=dt.ocr_content_key,
                )
            if not text:
                raise ValueError("No text content available")
            return text

    # ── Progress reporting ───────────────────────────────────────────

    @staticmethod
    def _pipeline_for_task(task_id: Optional[str]) -> Optional[str]:
        prefix = str(task_id or "").upper()
        if prefix.startswith("TRANSLATE"):
            return "translate"
        if prefix.startswith("EXTRACT"):
            return "extract"
        if prefix.startswith(("DIGEST_PIPELINE", "HIERARCHICAL_SUMMARIZE", "MAIN_CONTENT")):
            return "digest"
        return None

    def _progress(
        self,
        task_id: Optional[str],
        pct: int,
        msg: str,
        **structured,
    ) -> None:
        """
        Update task progress in the DB.  Safe to call with *task_id=None*
        (no-op), so callers never need to guard against missing task IDs.
        """
        if "pipeline" not in structured:
            pipeline = self._pipeline_for_task(task_id)
            if pipeline:
                structured["pipeline"] = pipeline
        if structured.get("pipeline") == "digest" and "stage" not in structured:
            prefix = str(task_id or "").upper()
            if prefix.startswith("HIERARCHICAL_SUMMARIZE"):
                structured["stage"] = "HIERARCHICAL_SUMMARIZE"
                structured.setdefault("mode", "standalone")
            elif prefix.startswith("MAIN_CONTENT"):
                structured["stage"] = "MAIN_CONTENT"
                structured.setdefault("mode", "standalone")
        emit_progress(task_id, pct, msg, **structured)

    # ── JSON extraction ──────────────────────────────────────────────

    def _parse_json_list(self, llm, response: str, list_key: str = None) -> tuple[list, bool]:
        """
        Parse a JSON list from an LLM response, reporting whether parsing failed.

        ``_extract_json`` collapses "the model found nothing" and "the response
        was not usable JSON" into the same empty list.  Callers that must tell
        those apart — because one is a legitimate result and the other is a
        truncated answer that should not overwrite stored data — use this.

        Returns:
            (items, parse_failed)
        """
        try:
            raw = llm.extract_json(response)
        except Exception as exc:
            logger.warning(
                "Failed to extract JSON from LLM response: %s | response snippet: %r",
                exc,
                response[:200],
            )
            return [], True

        if isinstance(raw, list):
            return raw, False
        if isinstance(raw, dict) and list_key and isinstance(raw.get(list_key), list):
            return raw[list_key], False

        logger.warning(
            "LLM response parsed as %s, expected a JSON list | response snippet: %r",
            type(raw).__name__,
            response[:200],
        )
        return [], True

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
        return self._parse_json_list(llm, response, list_key)[0]
