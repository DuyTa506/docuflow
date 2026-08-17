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
from data.database import get_db_manager
from data.db_models import DigitizedText, Document
from data.id_generator import IdGenerator
from services.base_service import BaseTaskService
from services.normalization_service import NormalizationService
from services.task_manager import TaskManager, task_manager


def _validate_pdf_readable(path: str) -> int:
    """Return the PDF's page count, rejecting files extraction cannot read.

    Raises ValueError with a user-facing (Vietnamese) message for
    password-protected or structurally unreadable PDFs.
    """
    import pymupdf

    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"Không đọc được file PDF (hỏng hoặc sai định dạng): {exc}") from exc
    try:
        if pdf.needs_pass:
            raise ValueError("PDF được bảo vệ bằng mật khẩu — hãy gỡ mật khẩu rồi tải lên lại.")
        if pdf.page_count <= 0:
            raise ValueError("PDF không có trang nào đọc được.")
        return pdf.page_count
    finally:
        pdf.close()


# How many text-layer pages Docling converts per call. Docling offers no
# per-page hook, so the slice size *is* the progress granularity: 761 pages at
# 10 gives 76 visible steps instead of one opaque wait, and each slice is a
# checkpoint a Temporal retry can resume from. Smaller would re-open the PDF
# more often for no extra signal a human can read.
DOCLING_PAGE_CHUNK = 10


def _page_windows(pages: list[int]) -> list[list[int]]:
    """Group ascending page numbers into windows of DOCLING_PAGE_CHUNK.

    The window bounds the *span* Docling converts, not the count we read
    back — which is the same thing when the pages are consecutive, and the
    thing that matters when they are not. Slicing the list ten at a time
    instead would turn a scattered set (a scanned book with occasional text
    pages, or a resume with holes) into a handful of near-whole-book
    conversions: worse than the single call this replaced.
    """
    windows: list[list[int]] = []
    for page in pages:
        if not windows or page - windows[-1][0] >= DOCLING_PAGE_CHUNK:
            windows.append([page])
        else:
            windows[-1].append(page)
    return windows


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

        # Count pages — and reject PDFs extraction can never read: silently
        # accepting them (total_pages=0) surfaced hours later as opaque
        # per-page OCR failures cascading into a FAILED digest (DOC_068,
        # password-protected upload).
        total_pages = 0
        if fmt == "pdf":
            total_pages = _validate_pdf_readable(file_path_on_disk)
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

    async def submit_extraction_async(
        self, db: Session, document_id: str, fairness_key: str = None
    ) -> tuple[str, bool]:
        """Temporal-aware extraction submit. With `ocr_use_temporal` off this
        is exactly `submit_extraction()`; on, it starts a durable
        ExtractionWorkflow whose retries resume from already-stored pages.
        Returns (task_id, reused).
        """
        if not settings.ocr_use_temporal:
            return self.submit_extraction(db, document_id)

        from data.db_models import Task
        from data.id_generator import IdGenerator
        from services.pipeline.temporal_client import start_extraction_workflow

        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc is None:
            raise ValueError("Document not found")

        active = (
            db.query(Task)
            .filter(
                Task.document_id == document_id,
                Task.task_type == "EXTRACT",
                Task.status.in_(["PENDING", "RUNNING"]),
            )
            .order_by(Task.created_at.desc())
            .first()
        )
        if active:
            return active.id, True

        raw_id = IdGenerator.next_id(db, "tasks")
        task_id = f"EXTRACT_{raw_id.split('_')[-1]}"
        db.add(
            Task(
                id=task_id,
                document_id=document_id,
                task_type="EXTRACT",
                status="PENDING",
                progress=0,
                message="Extraction workflow queued",
            )
        )
        db.commit()

        await start_extraction_workflow(
            document_id=document_id, parent_task_id=task_id, fairness_key=fairness_key
        )
        return task_id, False

    async def _run_extraction(
        self,
        document_id: str,
        task_id: Optional[str] = None,
        resume: bool = False,
        mark_failed_on_error: bool = True,
        attempt: int = 1,
    ):
        """
        Background coroutine: unified extraction pipeline.

        `mark_failed_on_error=False` (Temporal path): a failed attempt must
        NOT mark the document FAILED while the workflow will still retry —
        the digest/translation extraction gates treat FAILED as terminal, so
        a transient attempt-1 blip would permanently kill those pipelines.
        Terminal marking happens in fail_extraction_activity instead.

        Routes each file format through the appropriate extractor:
          - doc   → LibreOffice → DOCX → DocxExtractor
          - docx  → DocxExtractor
          - pdf   → per-page classify: text page → DoclingPdfExtractor,
                                       scanned   → OcrExtractor (vLLM)
          - image → OcrExtractor (vLLM)

        All paths produce UnifiedElement[] → layout_element dicts →
        saved to DB Pages + LayoutElements + DigitizedText.

        With `resume=True` (Temporal retry), existing extraction artifacts
        are KEPT and already-stored pages are skipped — a crash at page 650
        of a 700-page book re-OCRs only the missing 50.
        """
        db_manager = get_db_manager()

        with db_manager.session() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if not doc:
                raise ValueError(f"Document {document_id} not found")
            doc.processing_status = "EXTRACT_IN_PROGRESS"
            from data.repositories import DocumentRepository

            if not resume:
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
                resume=resume,
                attempt=attempt,
            )
        except Exception:
            if mark_failed_on_error:
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

    async def _extract_text_pages(
        self,
        *,
        file_path: str,
        pages: list[int],
        extractor,
        save_page,
        on_page_done,
    ) -> None:
        """Convert and persist the text-layer pages, a slice at a time.

        Docling converts in one blocking call with no per-page hook. Handed a
        whole 761-page book that call froze the activity's event loop for the
        best part of an hour: `_with_heartbeat` never got a slot, so Temporal
        killed the attempt on its 5-minute `heartbeat_timeout` and threw the
        finished work away — and the same frozen loop could not poll for the
        retry it had just scheduled, so the worker deadlocked. Nothing was
        persisted and no progress was reported in the meantime, which is why
        the task sat at PENDING 0% and read as dead.

        Slicing addresses each of those: `to_thread` keeps the loop free for
        the whole conversion, every slice persists its pages before the next
        one starts, and progress advances from the first slice on.
        """
        from utils.image_utils import render_pdf_page_to_base64

        for slice_pages in _page_windows(pages):
            page_range = (slice_pages[0], slice_pages[-1])

            def _convert_slice(page_range=page_range):
                extractor.convert(page_range=page_range)

            await asyncio.to_thread(_convert_slice)

            for page_num in slice_pages:

                def _read_and_render(page_num=page_num):
                    unified_elements = extractor.extract_page(page_num)
                    page_w, page_h = extractor.page_size(page_num)
                    page_markdown = extractor.page_markdown(page_num)
                    # 72 DPI raster: 1 px ≈ 1 PDF point so docling bboxes
                    # align with the page image.
                    page_image_b64 = render_pdf_page_to_base64(
                        file_path,
                        page_num,
                        target_dpi=72,
                        max_size=max(int(page_w), int(page_h), 4096),
                    )
                    return unified_elements, page_w, page_h, page_markdown, page_image_b64

                elements, page_w, page_h, page_markdown, page_image_b64 = await asyncio.to_thread(
                    _read_and_render
                )

                save_page(
                    page_num,
                    markdown_content=page_markdown,
                    layout_dicts=[e.to_layout_element_dict() for e in elements],
                    image_width=int(page_w),
                    image_height=int(page_h),
                    page_image_b64=page_image_b64,
                )
                on_page_done()

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
        resume: bool = False,
        attempt: int = 1,
    ):
        from openai import AsyncOpenAI

        from services.extractors.doc_converter import convert_doc_to_docx
        from services.extractors.docling_layout_extractor import DoclingLayoutExtractor
        from services.extractors.docling_pdf_extractor import DoclingPdfExtractor, classify_pages
        from services.extractors.docx_extractor import DocxExtractor
        from services.extractors.ocr_extractor import OcrExtractor
        from services.storage_service import DocumentStorageService

        all_markdown_parts = []
        element_count = 0
        resolved_mode = (
            "docx" if fmt in ("doc", "docx") else ("image" if fmt == "image" else "pdf_hybrid")
        )

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

            page_numbers = sorted(pages_map.keys())
            total_docx_pages = len(page_numbers)
            for processed_count, page_num in enumerate(page_numbers, start=1):
                page_elements = pages_map[page_num]
                layout_dicts = [e.to_layout_element_dict() for e in page_elements]
                page_markdown = "\n\n".join(e.text for e in page_elements if e.text)

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
                    pct = int((processed_count / max(1, total_docx_pages)) * 95)
                    with db_manager.session() as db:
                        TaskManager.update_progress(
                            db,
                            task_id,
                            pct,
                            f"Page {processed_count}/{total_docx_pages}",
                            {
                                "version": 1,
                                "pipeline": "extract",
                                "phase": "active",
                                "mode": "docx",
                                "unit_kind": "page",
                                "units_done": processed_count,
                                "units_total": total_docx_pages,
                                "attempt": attempt,
                            },
                        )
        # ── PDF path (hybrid per-page) ───────────────────────────────
        elif fmt == "pdf":
            # Say something before the first page lands. Classification parses
            # the whole PDF and the layout pass follows it, so on a long book
            # nothing reported for a long while — and `update_progress` is what
            # flips the task PENDING -> RUNNING, so the UI read "queued" the
            # entire time and looked dead.
            if task_id:
                with db_manager.session() as db:
                    TaskManager.update_progress(
                        db,
                        task_id,
                        0,
                        f"Classifying {total_pages} page(s)",
                        {
                            "version": 1,
                            "pipeline": "extract",
                            "phase": "classifying",
                            "mode": "pdf_hybrid",
                            "unit_kind": "page",
                            "units_done": 0,
                            "units_total": total_pages,
                            "attempt": attempt,
                        },
                    )

            def _classify():
                page_classifier = DoclingPdfExtractor(file_path)
                return classify_pages(page_classifier._doc, threshold=settings.pdf_text_threshold)

            # Off the event loop: this parses every page, and the activity's
            # heartbeat is an asyncio task on that same loop.
            page_types = await asyncio.to_thread(_classify)

            # Pages persisted by a previous (crashed) attempt — skip them so a
            # Temporal retry resumes instead of re-OCRing the whole book.
            done_pages: set[int] = set()
            if resume:
                with db_manager.session() as db:
                    from data.db_models import Page

                    done_pages = {
                        row[0]
                        for row in db.query(Page.page_number)
                        .filter(Page.document_id == document_id)
                        .all()
                    }
                if done_pages:
                    import logging

                    logging.getLogger(__name__).info(
                        "Extraction resume for %s: %d/%d page(s) already stored",
                        document_id,
                        len(done_pages),
                        total_pages,
                    )

            pending_text_pages = [
                p
                for p in range(1, total_pages + 1)
                if page_types.get(p, "scanned") == "text" and p not in done_pages
            ]
            scanned_pages = [
                p
                for p in range(1, total_pages + 1)
                if page_types.get(p, "scanned") != "text" and p not in done_pages
            ]
            scanned_total = sum(
                1 for p in range(1, total_pages + 1) if page_types.get(p, "scanned") != "text"
            )
            if scanned_total == 0:
                extraction_mode = "pdf_text"
            elif scanned_total == total_pages:
                extraction_mode = "pdf_scan"
            else:
                extraction_mode = "pdf_hybrid"
            resolved_mode = extraction_mode
            if task_id:
                with db_manager.session() as db:
                    TaskManager.update_progress(
                        db,
                        task_id,
                        int((len(done_pages) / max(1, total_pages)) * 95),
                        f"Page {len(done_pages)}/{total_pages}",
                        {
                            "version": 1,
                            "pipeline": "extract",
                            "phase": "active",
                            "mode": extraction_mode,
                            "unit_kind": "page",
                            "units_done": len(done_pages),
                            "units_total": total_pages,
                            "attempt": attempt,
                            "checkpoint_units": len(done_pages),
                        },
                    )

            client = AsyncOpenAI(
                api_key=settings.vllm_api_key,
                base_url=settings.vllm_server_url,
            )

            done_counter = [len(done_pages)]

            def _bump_progress() -> None:
                done_counter[0] += 1
                if task_id:
                    pct = int((done_counter[0] / total_pages) * 100)
                    with db_manager.session() as db:
                        TaskManager.update_progress(
                            db,
                            task_id,
                            pct,
                            f"Page {done_counter[0]}/{total_pages}",
                            {
                                "version": 1,
                                "pipeline": "extract",
                                "phase": "active",
                                "mode": extraction_mode,
                                "unit_kind": "page",
                                "units_done": done_counter[0],
                                "units_total": total_pages,
                                "attempt": attempt,
                                "checkpoint_units": len(done_pages),
                            },
                        )

            # Text-layer pages (cheap, no LLM) — persisted page by page.
            if pending_text_pages:

                def _save_text_page(page_number: int, **fields) -> None:
                    with db_manager.session() as db:
                        DocumentStorageService(db).save_unified_elements(
                            document_id=document_id,
                            page_number=page_number,
                            page_type="text",
                            **fields,
                        )

                await self._extract_text_pages(
                    file_path=file_path,
                    pages=pending_text_pages,
                    extractor=DoclingLayoutExtractor(file_path),
                    save_page=_save_text_page,
                    on_page_done=_bump_progress,
                )

            async def _extract_scanned(_idx: int, page_num: int):
                # Fresh OcrExtractor per page: extract_page() stashes its raw
                # result on `self.page_result`, so sharing one instance across
                # concurrent calls would race.
                extractor = OcrExtractor(client, file_path)
                unified_elements = await extractor.extract_page(page_num)
                page_result = extractor.page_result

                # Persist IMMEDIATELY — each stored page is a checkpoint. The
                # old design held every OCR result in memory until all pages
                # finished, so a crash at page 650 lost all 650.
                if page_result is not None:
                    with db_manager.session() as db:
                        DocumentStorageService(db).save_page_result(
                            document_id, page_result, page_type="scanned"
                        )
                elif unified_elements:
                    layout_dicts = [e.to_layout_element_dict() for e in unified_elements]
                    page_markdown = "\n\n".join(e.text for e in unified_elements if e.text)
                    with db_manager.session() as db:
                        DocumentStorageService(db).save_unified_elements(
                            document_id=document_id,
                            page_number=page_num,
                            markdown_content=page_markdown,
                            layout_dicts=layout_dicts,
                            page_type="scanned",
                        )
                _bump_progress()
                return page_num

            from services.translators._parallel import run_parallel

            await run_parallel(
                scanned_pages,
                _extract_scanned,
                parallelism=settings.ocr_page_parallelism,
            )

            # Assemble from the DB in page order — uniform for fresh runs and
            # resumes (skipped pages' markdown lives only in the DB).
            with db_manager.session() as db:
                from data.db_models import Page

                rows = (
                    db.query(Page.page_number, Page.markdown_content)
                    .filter(Page.document_id == document_id)
                    .order_by(Page.page_number)
                    .all()
                )
                all_markdown_parts = [row[1] or "" for row in rows]
                from data.repositories import DocumentRepository

                element_count = DocumentRepository(db).count_elements(document_id)

        # ── Image path ───────────────────────────────────────────────
        elif fmt == "image":
            client = AsyncOpenAI(
                api_key=settings.vllm_api_key,
                base_url=settings.vllm_server_url,
            )
            ocr_extractor = OcrExtractor(client, file_path)
            if task_id:
                with db_manager.session() as db:
                    TaskManager.update_progress(
                        db,
                        task_id,
                        0,
                        "OCR image",
                        {
                            "version": 1,
                            "pipeline": "extract",
                            "phase": "active",
                            "mode": "image",
                            "unit_kind": "page",
                            "units_done": 0,
                            "units_total": 1,
                            "attempt": attempt,
                        },
                    )
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
                    TaskManager.update_progress(
                        db,
                        task_id,
                        95,
                        "OCR page done",
                        {
                            "version": 1,
                            "pipeline": "extract",
                            "phase": "active",
                            "mode": "image",
                            "unit_kind": "page",
                            "units_done": 1,
                            "units_total": 1,
                            "attempt": attempt,
                        },
                    )

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

        ocr_inline, ocr_key = maybe_offload_text(document_id, field="ocr", content=full_text)
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

            if settings.stage_rerun_use_temporal:
                from services.stage_dispatch import submit_stage

                await submit_stage(
                    db,
                    document_id,
                    "BUILD_TREE",
                    options={"use_spatial_metadata": True, "if_add_node_summary": "no"},
                )
            else:
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
                    TaskManager.update_progress(
                        db2,
                        task_id,
                        98,
                        "Preparing DOCX & PDF exports…",
                        {
                            "version": 1,
                            "pipeline": "extract",
                            "phase": "exporting",
                            "mode": resolved_mode,
                            "unit_kind": "export",
                            "units_done": 0,
                            "units_total": 1,
                            "attempt": attempt,
                        },
                    )
                await export_service.cache_ocr_exports_after_extract(db2, document_id)
                if task_id:
                    TaskManager.update_progress(
                        db2,
                        task_id,
                        100,
                        "Done",
                        {
                            "version": 1,
                            "pipeline": "extract",
                            "phase": "exporting",
                            "mode": resolved_mode,
                            "unit_kind": "export",
                            "units_done": 1,
                            "units_total": 1,
                            "attempt": attempt,
                        },
                    )

        await _cache_exports_after_extract()

        return {"pages_processed": total_pages, "element_count": element_count}
