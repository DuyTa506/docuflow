"""
Document management service.

Handles: file upload, OCR orchestration, normalization orchestration,
and the new unified extraction pipeline (DOCX / DOC / PDF text+OCR hybrid).
"""
import os
import shutil
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from config.settings import settings
from data.db_models import Document, Page, LayoutElement, DigitizedText
from data.database import get_db_manager
from data.id_generator import IdGenerator
from services.base_service import BaseTaskService
from services.task_manager import task_manager, TaskManager
from services.normalization_service import NormalizationService


class DocumentService(BaseTaskService):
    """High-level document operations (upload, trigger OCR, trigger normalization)."""

    def upload_document(
        self,
        db: Session,
        file_path_on_disk: str,
        original_filename: str,
        user_id: Optional[str],
        title: Optional[str] = None,
        source_language: str = "en",
    ) -> Document:
        """
        Register a new document. The file has already been written to
        *file_path_on_disk* by the router.
        """
        # Detect format
        ext = os.path.splitext(original_filename)[1].lower()
        fmt_map = {
            ".pdf": "pdf",
            ".png": "image",
            ".jpg": "image",
            ".jpeg": "image",
            ".docx": "docx",
            ".doc": "doc",
        }
        fmt = fmt_map.get(ext, "unknown")
        file_type = "pdf" if fmt == "pdf" else ("image" if fmt == "image" else fmt)

        # Count pages
        total_pages = 0
        if fmt == "pdf":
            try:
                from PyPDF2 import PdfReader
                total_pages = len(PdfReader(file_path_on_disk).pages)
            except Exception:
                total_pages = 0
        elif fmt == "image":
            total_pages = 1
        elif fmt in ("docx", "doc"):
            total_pages = 1  # logical count; actual pages resolved during extraction

        # Auto-detect language if not provided
        if not source_language or source_language == "auto":
            source_language = "en"  # default fallback

        doc_id = IdGenerator.next_id(db, "documents")
        doc = Document(
            id=doc_id,
            user_id=user_id,
            title=title or original_filename,
            original_filename=original_filename,
            source_language=source_language,
            format=fmt,
            file_path=file_path_on_disk,
            file_type=file_type,
            total_pages=total_pages,
            processing_status="INIT",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc

    # ── Unified Extraction background task ──────────────────────────

    def submit_extraction(self, db: Session, document_id: str) -> str:
        """
        Submit a unified extraction background task (DOCX / DOC / PDF hybrid).
        Returns the task_id.
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            raise ValueError("Document not found")

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="EXTRACT",
            coro=self._run_extraction(document_id),
        )
        return task_id

    async def _run_extraction(self, document_id: str):
        """
        Background coroutine: unified extraction pipeline.

        Routes each file format through the appropriate extractor:
          - doc   → LibreOffice → DOCX → DocxExtractor
          - docx  → DocxExtractor
          - pdf   → per-page classify: text page → PdfTextExtractor,
                                       scanned   → OcrExtractor (vLLM)
          - image → OcrExtractor (vLLM)

        All paths produce UnifiedElement[] → layout_element dicts →
        saved to DB Pages + LayoutElements + DigitizedText.
        """
        from openai import AsyncOpenAI
        from services.extractors.docx_extractor import DocxExtractor
        from services.extractors.pdf_text_extractor import PdfTextExtractor, classify_pages
        from services.extractors.ocr_extractor import OcrExtractor, ocr_elements_to_unified
        from services.extractors.doc_converter import convert_doc_to_docx
        from serving.storage_service import DocumentStorageService

        db_manager = get_db_manager()

        with db_manager.session() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                raise ValueError(f"Document {document_id} not found")
            doc.processing_status = "EXTRACT_IN_PROGRESS"
            file_path = doc.file_path
            fmt = doc.format or ""
            total_pages = doc.total_pages or 1

        # Find task for progress reporting
        task_id = self._find_task_id(document_id, "EXTRACT")

        all_markdown_parts = []
        element_count = 0

        # ── DOCX / DOC path ─────────────────────────────────────────
        if fmt in ("docx", "doc"):
            if fmt == "doc":
                docx_path = convert_doc_to_docx(
                    file_path,
                    libreoffice_path=settings.libreoffice_path,
                )
            else:
                docx_path = file_path

            extractor = DocxExtractor()
            unified_elements = extractor.extract(docx_path)

            # Group elements by page_number for DB storage
            pages_map: dict = {}
            for elem in unified_elements:
                pages_map.setdefault(elem.page_number, []).append(elem)

            for page_num in sorted(pages_map.keys()):
                page_elements = pages_map[page_num]
                layout_dicts = [e.to_layout_element_dict() for e in page_elements]
                page_markdown = "\n\n".join(
                    e.text for e in page_elements if e.text
                )

                with db_manager.session() as db:
                    storage = DocumentStorageService(db)
                    storage.save_unified_elements(
                        document_id=document_id,
                        page_number=page_num,
                        markdown_content=page_markdown,
                        layout_dicts=layout_dicts,
                    )

                all_markdown_parts.append(page_markdown)
                element_count += len(layout_dicts)

                if task_id:
                    pct = int((page_num / max(pages_map.keys())) * 100)
                    with db_manager.session() as db:
                        TaskManager.update_progress(db, task_id, pct, f"Page {page_num}")
        # ── PDF path (hybrid per-page) ───────────────────────────────
        elif fmt == "pdf":
            page_types = classify_pages(file_path, threshold=settings.pdf_text_threshold)

            client = AsyncOpenAI(
                api_key=settings.vllm_api_key,
                base_url=settings.vllm_server_url,
            )
            pdf_extractor = PdfTextExtractor(file_path)
            ocr_extractor = OcrExtractor(client, file_path)

            for page_num in range(1, total_pages + 1):
                page_type = page_types.get(page_num, "scanned")

                if page_type == "text":
                    # Direct text extraction
                    unified_elements = pdf_extractor.extract_page(page_num)
                    layout_dicts = [e.to_layout_element_dict() for e in unified_elements]
                    page_markdown = "\n\n".join(
                        e.text for e in unified_elements if e.text
                    )

                    with db_manager.session() as db:
                        storage = DocumentStorageService(db)
                        storage.save_unified_elements(
                            document_id=document_id,
                            page_number=page_num,
                            markdown_content=page_markdown,
                            layout_dicts=layout_dicts,
                        )
                    all_markdown_parts.append(page_markdown)
                    element_count += len(layout_dicts)

                else:
                    # OCR path (scanned page) — reuse existing logic
                    result = await ocr_extractor.extract_page(page_num)
                    if isinstance(result, tuple):
                        unified_elements, page_result = result
                    else:
                        unified_elements, page_result = result, None

                    if page_result is not None:
                        with db_manager.session() as db:
                            storage = DocumentStorageService(db)
                            storage.save_page_result(document_id, page_result)
                        all_markdown_parts.append(page_result.markdown or "")
                        element_count += len(page_result.layout_elements or [])
                    elif unified_elements:
                        layout_dicts = [e.to_layout_element_dict() for e in unified_elements]
                        page_markdown = "\n\n".join(e.text for e in unified_elements if e.text)
                        with db_manager.session() as db:
                            storage = DocumentStorageService(db)
                            storage.save_unified_elements(
                                document_id=document_id,
                                page_number=page_num,
                                markdown_content=page_markdown,
                                layout_dicts=layout_dicts,
                            )
                        all_markdown_parts.append(page_markdown)
                        element_count += len(layout_dicts)

                if task_id:
                    pct = int((page_num / total_pages) * 100)
                    with db_manager.session() as db:
                        TaskManager.update_progress(db, task_id, pct, f"Page {page_num}/{total_pages}")

        # ── Image path ───────────────────────────────────────────────
        elif fmt == "image":
            client = AsyncOpenAI(
                api_key=settings.vllm_api_key,
                base_url=settings.vllm_server_url,
            )
            ocr_extractor = OcrExtractor(client, file_path)
            result = await ocr_extractor.extract_page(1)
            if isinstance(result, tuple):
                unified_elements, page_result = result
            else:
                unified_elements, page_result = result, None

            if page_result is not None:
                with db_manager.session() as db:
                    storage = DocumentStorageService(db)
                    storage.save_page_result(document_id, page_result)
                all_markdown_parts.append(page_result.markdown or "")
                element_count += len(page_result.layout_elements or [])

            if task_id:
                with db_manager.session() as db:
                    TaskManager.update_progress(db, task_id, 100, "Done")

        # ── Save aggregated text ─────────────────────────────────────
        full_text = "\n\n---\n\n".join(all_markdown_parts)
        with db_manager.session() as db:
            from data.db_models import DigitizedText
            dt = DigitizedText(document_id=document_id, ocr_content=full_text)
            db.add(dt)
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.processing_status = "EXTRACTED"

        return {"pages_processed": total_pages, "element_count": element_count}

    # ── OCR background task ─────────────────────────────────────────

    def submit_ocr(self, db: Session, document_id: str) -> str:
        """Submit an OCR background task. Returns the task_id."""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            raise ValueError("Document not found")

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="OCR",
            coro=self._run_ocr(document_id),
        )
        return task_id

    async def _run_ocr(self, document_id: str):
        """
        Background coroutine: run OCR on every page of the document.
        """
        from openai import AsyncOpenAI
        from serving.logic import process_page_api

        db_manager = get_db_manager()

        with db_manager.session() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                raise ValueError(f"Document {document_id} not found")
            doc.processing_status = "OCR_IN_PROGRESS"
            file_path = doc.file_path
            total_pages = doc.total_pages or 1

        # Find the task_id for progress reporting
        task_id = self._find_task_id(document_id, "OCR")

        client = AsyncOpenAI(
            api_key=settings.vllm_api_key,
            base_url=settings.vllm_server_url,
        )

        all_markdown_parts = []
        element_count = 0

        for page_num in range(1, total_pages + 1):
            page_result = None
            async for event in process_page_api(
                client=client,
                pdf_path=file_path,
                page_num=page_num,
                stream_enabled=False,
            ):
                if event.get("type") == "result":
                    page_result = event["result"]

            if page_result is None:
                continue

            # Save page to DB
            with db_manager.session() as db:
                from serving.storage_service import DocumentStorageService
                storage = DocumentStorageService(db)
                storage.save_page_result(document_id, page_result)

            all_markdown_parts.append(page_result.markdown or "")
            if page_result.layout_elements:
                element_count += len(page_result.layout_elements)

            # Report progress
            if task_id:
                pct = int((page_num / total_pages) * 100)
                with db_manager.session() as db:
                    TaskManager.update_progress(db, task_id, pct, f"Page {page_num}/{total_pages}")

        # Save aggregated OCR content
        full_ocr = "\n\n---\n\n".join(all_markdown_parts)
        with db_manager.session() as db:
            dt = DigitizedText(document_id=document_id, ocr_content=full_ocr)
            db.add(dt)
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.processing_status = "OCR_COMPLETED"

        return {"pages_processed": total_pages, "element_count": element_count}

    # ── Normalization background task ───────────────────────────────

    def submit_normalization(self, db: Session, document_id: str) -> str:
        """Submit a normalization background task. Returns the task_id."""
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            raise ValueError("Document not found")

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="NORMALIZE",
            coro=self._run_normalization(document_id),
        )
        return task_id

    async def _run_normalization(self, document_id: str):
        """
        Background coroutine: normalise the OCR text for a document.
        """
        db_manager = get_db_manager()
        normalizer = NormalizationService()

        with db_manager.session() as db:
            dt = (
                db.query(DigitizedText)
                .filter(DigitizedText.document_id == document_id)
                .first()
            )
            if dt is None or not dt.ocr_content:
                raise ValueError("No OCR content found — run OCR first")

            doc = db.query(Document).filter(Document.id == document_id).first()
            language = doc.source_language if doc else "en"
            ocr_text = dt.ocr_content

        normalized = normalizer.normalize(ocr_text, language)

        with db_manager.session() as db:
            dt = (
                db.query(DigitizedText)
                .filter(DigitizedText.document_id == document_id)
                .first()
            )
            if dt:
                dt.normalized_content = normalized
                dt.updated_at = datetime.utcnow()
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.processing_status = "NORMALIZED"

        return {"characters": len(normalized)}
