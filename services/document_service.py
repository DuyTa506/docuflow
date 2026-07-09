"""
Document management service.

Handles: file upload and unified extraction pipeline (extract + normalize in one step).
"""
import asyncio
import mimetypes
import os
from typing import Optional

from sqlalchemy.orm import Session

from config.settings import settings
from data.db_models import Document, DigitizedText
from data.database import get_db_manager
from data.id_generator import IdGenerator
from services.base_service import BaseTaskService
from services.task_manager import task_manager, TaskManager
from services.normalization_service import NormalizationService


class DocumentService(BaseTaskService):
    """High-level document operations (upload, trigger extraction, trigger normalization)."""

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
        from config.settings import normalize_lang_code

        source_language = normalize_lang_code(source_language or "en")

        doc_id = IdGenerator.next_id(db, "documents")
        safe_name = os.path.basename(original_filename).replace(" ", "_")

        from services.object_storage import get_object_storage
        from utils.storage_keys import original_key

        storage = get_object_storage()
        object_key = original_key(doc_id, safe_name)
        with open(file_path_on_disk, "rb") as src:
            file_bytes = src.read()
        content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        storage.put_bytes(object_key, file_bytes, content_type=content_type)
        try:
            os.remove(file_path_on_disk)
        except OSError:
            pass
        file_path_on_disk = object_key

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

    def submit_extraction(self, db: Session, document_id: str) -> tuple[str, bool]:
        """
        Submit a unified extraction background task (DOCX / DOC / PDF hybrid).
        Returns the task_id.
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            raise ValueError("Document not found")

        existing = task_manager.get_active_task_id(db, document_id, "EXTRACT")
        if existing:
            return existing, True

        task_id = task_manager.submit(
            db,
            document_id=document_id,
            task_type="EXTRACT",
            coro_factory=lambda tid: self._run_extraction(document_id, tid),
        )
        return task_id, False

    async def _run_extraction(self, document_id: str, task_id: Optional[str] = None):
        """
        Background coroutine: unified extraction pipeline.

        Routes each file format through the appropriate extractor:
          - doc   → LibreOffice → DOCX → DocxExtractor
          - docx  → DocxExtractor
          - pdf   → per-page classify: text page → DoclingPdfExtractor,
                                       scanned   → OcrExtractor (vLLM)
          - image → OcrExtractor (vLLM)

        All paths produce UnifiedElement[] → layout_element dicts →
        saved to DB Pages + LayoutElements + DigitizedText.
        """
        from openai import AsyncOpenAI
        from services.extractors.docx_extractor import DocxExtractor
        from services.extractors.docling_pdf_extractor import DoclingPdfExtractor, classify_pages
        from services.extractors.ocr_extractor import OcrExtractor, ocr_elements_to_unified
        from services.extractors.doc_converter import convert_doc_to_docx
        from services.storage_service import DocumentStorageService

        db_manager = get_db_manager()

        with db_manager.session() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                raise ValueError(f"Document {document_id} not found")
            doc.processing_status = "EXTRACT_IN_PROGRESS"
            from data.repositories import DocumentRepository

            DocumentRepository(db).clear_extraction_artifacts(document_id)
            from services.export_service import export_service

            export_service.invalidate_ocr_exports(document_id)
            file_path = doc.file_path
            fmt = doc.format or ""
            total_pages = doc.total_pages or 1

        local_path: str | None = None
        try:
            from services.object_storage import get_object_storage

            storage = get_object_storage()
            local_path = storage.resolve_local_or_key(file_path)
            return await self._run_extraction_body(
                document_id,
                file_path=local_path,
                storage_key=file_path,
                fmt=fmt,
                total_pages=total_pages,
                task_id=task_id,
                db_manager=db_manager,
            )
        except Exception:
            with db_manager.session() as db:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    doc.processing_status = "FAILED"
            raise
        finally:
            if local_path and local_path != file_path and os.path.isfile(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass

    async def _run_extraction_body(
        self,
        document_id: str,
        *,
        file_path: str,
        storage_key: str | None = None,
        fmt: str,
        total_pages: int,
        task_id: Optional[str],
        db_manager,
    ):
        from openai import AsyncOpenAI
        from services.extractors.docx_extractor import DocxExtractor
        from services.extractors.docling_pdf_extractor import DoclingPdfExtractor, classify_pages
        from services.extractors.docling_layout_extractor import DoclingLayoutExtractor
        from services.extractors.ocr_extractor import OcrExtractor
        from services.extractors.doc_converter import convert_doc_to_docx
        from services.storage_service import DocumentStorageService

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
            page_classifier = DoclingPdfExtractor(file_path)
            page_types = classify_pages(
                page_classifier._doc, threshold=settings.pdf_text_threshold
            )

            layout_extractor: DoclingLayoutExtractor | None = None
            if any(t == "text" for t in page_types.values()):
                layout_extractor = DoclingLayoutExtractor(file_path)
                layout_extractor.convert()

            client = AsyncOpenAI(
                api_key=settings.vllm_api_key,
                base_url=settings.vllm_server_url,
            )

            scanned_pages = [
                p for p in range(1, total_pages + 1)
                if page_types.get(p, "scanned") != "text"
            ]

            async def _extract_scanned(_idx: int, page_num: int):
                # Fresh OcrExtractor per page: extract_page() stashes its raw
                # result on `self.page_result`, so sharing one instance across
                # concurrent calls would race.
                extractor = OcrExtractor(client, file_path)
                unified_elements = await extractor.extract_page(page_num)
                return page_num, unified_elements, extractor.page_result

            from services.translators._parallel import run_parallel

            scanned_results = {
                page_num: (unified_elements, page_result)
                for page_num, unified_elements, page_result in await run_parallel(
                    scanned_pages,
                    _extract_scanned,
                    parallelism=settings.ocr_page_parallelism,
                )
            }

            for page_num in range(1, total_pages + 1):
                page_type = page_types.get(page_num, "scanned")

                if page_type == "text":
                    assert layout_extractor is not None
                    unified_elements = layout_extractor.extract_page(page_num)
                    page_w, page_h = layout_extractor.page_size(page_num)
                    page_markdown = layout_extractor.page_markdown(page_num)

                    layout_dicts = [e.to_layout_element_dict() for e in unified_elements]

                    # 72 DPI raster: 1 px ≈ 1 PDF point so docling bboxes align with page image.
                    from utils.image_utils import render_pdf_page_to_base64

                    page_image_b64 = render_pdf_page_to_base64(
                        file_path,
                        page_num,
                        target_dpi=72,
                        max_size=max(int(page_w), int(page_h), 4096),
                    )

                    with db_manager.session() as db:
                        storage = DocumentStorageService(db)
                        storage.save_unified_elements(
                            document_id=document_id,
                            page_number=page_num,
                            markdown_content=page_markdown,
                            layout_dicts=layout_dicts,
                            page_type="text",
                            image_width=int(page_w),
                            image_height=int(page_h),
                            page_image_b64=page_image_b64,
                        )
                    all_markdown_parts.append(page_markdown)
                    element_count += len(layout_dicts)

                else:
                    # OCR path (scanned page) — extracted concurrently above.
                    unified_elements, page_result = scanned_results[page_num]

                    if page_result is not None:
                        with db_manager.session() as db:
                            storage = DocumentStorageService(db)
                            storage.save_page_result(document_id, page_result, page_type="scanned")
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
                                page_type="scanned",
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
            unified_elements = await ocr_extractor.extract_page(1)
            page_result = ocr_extractor.page_result

            if page_result is not None:
                with db_manager.session() as db:
                    storage = DocumentStorageService(db)
                    storage.save_page_result(document_id, page_result, page_type="scanned")
                all_markdown_parts.append(page_result.markdown or "")
                element_count += len(page_result.layout_elements or [])

            if task_id:
                with db_manager.session() as db:
                    TaskManager.update_progress(db, task_id, 95, "OCR page done")

        # ── Normalize + save aggregated text ────────────────────────
        full_text = "\n\n---\n\n".join(all_markdown_parts)

        with db_manager.session() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            fallback_language = doc.source_language if doc else "en"

        from utils.lang_detect import detect_source_language, sample_representative_text

        lang_sample = sample_representative_text(all_markdown_parts)
        language = detect_source_language(lang_sample, fallback=fallback_language)

        normalized_text = NormalizationService().normalize(full_text, language)

        from utils.content_storage import maybe_offload_text

        ocr_inline, ocr_key = maybe_offload_text(
            document_id, field="ocr", content=full_text
        )
        norm_inline, norm_key = maybe_offload_text(
            document_id, field="normalized", content=normalized_text
        )

        with db_manager.session() as db:
            dt = DigitizedText(
                document_id=document_id,
                ocr_content=ocr_inline,
                ocr_content_key=ocr_key,
                normalized_content=norm_inline,
                normalized_content_key=norm_key,
            )
            db.add(dt)
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.processing_status = "EXTRACTED"
                doc.source_language = language

        # ── Auto-submit tree index build (non-blocking) ─────────────
        with db_manager.session() as db:
            from serving.tree_indexing_service import TreeIndexingService

            async def _auto_build_tree():
                dbm = get_db_manager()
                with dbm.session() as db2:
                    svc = TreeIndexingService(db2)
                    return await svc.build_enhanced_tree_index(
                        document_id=document_id,
                        use_spatial_metadata=True,
                        if_add_node_summary="no",
                    )

            task_manager.submit(
                db,
                document_id=document_id,
                task_type="BUILD_TREE",
                coro=_auto_build_tree(),
            )

        async def _cache_exports_after_extract() -> None:
            dbm = get_db_manager()
            with dbm.session() as db2:
                from services.export_service import export_service

                if task_id:
                    TaskManager.update_progress(db2, task_id, 98, "Preparing DOCX & PDF exports…")
                await export_service.cache_ocr_exports_after_extract(db2, document_id)
                if task_id:
                    TaskManager.update_progress(db2, task_id, 100, "Done")

        await _cache_exports_after_extract()

        return {"pages_processed": total_pages, "element_count": element_count}
