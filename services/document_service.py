"""
Document management service.

Handles: file upload and unified extraction pipeline (extract + normalize in one step).
"""
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


_SMALL_IMAGE_PX = 25  # skip images whose largest side < 25 px (decorative dots / tiny icon fragments)


async def _ocr_embedded_image(client, img_b64: str, width_px: int = 0, height_px: int = 0) -> str:
    """
    Describe or transcribe an embedded image via the vLLM OCR server.

    Small images (icons, logos, decorative) are skipped — OCR on them produces
    hallucinated descriptions.  Larger images get a prompt chosen by aspect ratio:
    wide images are treated as figures/charts; squarish images as photos/illustrations.

    Returns an empty string on any failure or when the image is too small.
    """
    if width_px < _SMALL_IMAGE_PX and height_px < _SMALL_IMAGE_PX:
        return ""  # icon / logo — skip OCR, keep raw placeholder

    # Wide → figure/chart/diagram.  Square-ish → photo/illustration.
    aspect = width_px / max(height_px, 1)
    if aspect > 1.3:
        prompt = "<image>\nParse the figure."
    else:
        prompt = "<image>\nDescribe this image in detail."

    try:
        response = await client.chat.completions.create(
            model=settings.vllm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            }],
            max_tokens=512,
            temperature=0.0,
            extra_body={
                "skip_special_tokens": False,
                "logits_processors": [
                    {
                        "qualname": "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
                        "kwargs": {
                            "ngram_size": 20,
                            "window_size": 50,
                            "whitelist_token_ids": [128821, 128822],
                        },
                    }
                ],
            },
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""


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
            pdf_extractor = DoclingPdfExtractor(file_path)
            page_types = classify_pages(pdf_extractor._doc, threshold=settings.pdf_text_threshold)

            client = AsyncOpenAI(
                api_key=settings.vllm_api_key,
                base_url=settings.vllm_server_url,
            )
            ocr_extractor = OcrExtractor(client, file_path)

            for page_num in range(1, total_pages + 1):
                page_type = page_types.get(page_num, "scanned")

                if page_type == "text":
                    # Direct text extraction
                    unified_elements = pdf_extractor.extract_page(page_num)

                    # OCR embedded image blocks using the vLLM server
                    for elem in unified_elements:
                        if elem.element_type == "image" and elem.image_bytes_b64:
                            ocr_text = await _ocr_embedded_image(
                                client, elem.image_bytes_b64,
                                elem.image_width or 0, elem.image_height or 0,
                            )
                            if ocr_text:
                                elem.text = f"[Image: {ocr_text}]"

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
                    # OCR path (scanned page) — convert PDF page to image → DeepSeek
                    unified_elements = await ocr_extractor.extract_page(page_num)
                    page_result = ocr_extractor.page_result  # set by extract_page()

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
            unified_elements = await ocr_extractor.extract_page(1)
            page_result = ocr_extractor.page_result

            if page_result is not None:
                with db_manager.session() as db:
                    storage = DocumentStorageService(db)
                    storage.save_page_result(document_id, page_result)
                all_markdown_parts.append(page_result.markdown or "")
                element_count += len(page_result.layout_elements or [])

            if task_id:
                with db_manager.session() as db:
                    TaskManager.update_progress(db, task_id, 100, "Done")

        # ── Normalize + save aggregated text ────────────────────────
        full_text = "\n\n---\n\n".join(all_markdown_parts)

        with db_manager.session() as db:
            doc = db.query(Document).filter(Document.id == document_id).first()
            language = doc.source_language if doc else "en"

        normalized_text = NormalizationService().normalize(full_text, language)

        with db_manager.session() as db:
            dt = DigitizedText(
                document_id=document_id,
                ocr_content=full_text,
                normalized_content=normalized_text,
            )
            db.add(dt)
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.processing_status = "EXTRACTED"

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

        return {"pages_processed": total_pages, "element_count": element_count}
