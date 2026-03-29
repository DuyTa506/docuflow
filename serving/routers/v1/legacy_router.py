"""
V1 legacy endpoints — preserved for backward compatibility.

These are the original endpoints from serving/workflow_api.py, extracted
into their own router so workflow_api.py can be a thin shell.

Endpoints:
  POST /process-document
  POST /build-index/{document_id}
  GET  /documents
  GET  /documents/{document_id}
  GET  /documents/{document_id}/markdown
  GET  /documents/{document_id}/elements
  GET  /documents/{document_id}/tree
"""
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from api.schemas import TreeIndexRequest
from api.dependencies import get_db
from data.db_models import Page
from serving.storage_service import DocumentStorageService
from serving.tree_indexing_service import TreeIndexingService
from serving.logic import process_page_api

# Configuration from environment (v1 uses its own client — preserved as-is)
_API_KEY = os.getenv("VLLM_API_KEY", "123")
_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1")

router = APIRouter(tags=["v1-legacy"])


@router.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    store_to_db: bool = Query(True, description="Store results to database"),
    db: Session = Depends(get_db),
):
    """
    Process PDF/image through OCR and optionally store to database.
    """
    from openai import AsyncOpenAI

    temp_fd, temp_path = tempfile.mkstemp(
        suffix=os.path.splitext(file.filename)[1]
    )
    try:
        content = await file.read()
        with os.fdopen(temp_fd, "wb") as f:
            f.write(content)

        file_type = "pdf" if file.filename.lower().endswith(".pdf") else "image"

        if file_type == "pdf":
            from PyPDF2 import PdfReader
            reader = PdfReader(temp_path)
            num_pages = len(reader.pages)
        else:
            num_pages = 1

        storage = DocumentStorageService(db)
        document = None
        if store_to_db:
            document = storage.create_document(
                filename=file.filename,
                file_type=file_type,
                total_pages=num_pages,
            )

        client = AsyncOpenAI(api_key=_API_KEY, base_url=_SERVER_URL)

        element_count = 0
        for page_num in range(1, num_pages + 1):
            page_result = None
            async for event in process_page_api(
                client=client,
                pdf_path=temp_path,
                page_num=page_num,
                stream_enabled=False,
            ):
                if event.get("type") == "result":
                    page_result = event["result"]

            if store_to_db and page_result and document:
                storage.save_page_result(document.id, page_result)
                if page_result.layout_elements:
                    element_count += len(page_result.layout_elements)

        return {
            "document_id": document.id if document else None,
            "filename": file.filename,
            "file_type": file_type,
            "total_pages": num_pages,
            "element_count": element_count,
            "stored_to_db": store_to_db,
        }

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post("/build-index/{document_id}")
async def build_index(
    document_id: str,
    request: TreeIndexRequest,
    db: Session = Depends(get_db),
):
    """Build PageIndex tree structure from stored document."""
    tree_service = TreeIndexingService(
        session=db,
        llm_provider=request.llm_provider,
        model=request.model,
    )
    try:
        result = await tree_service.build_enhanced_tree_index(
            document_id=document_id,
            use_spatial_metadata=request.use_spatial_metadata,
            discover_implicit_sections=request.discover_implicit_sections,
            spatial_weights=request.spatial_weights,
            if_thinning=request.if_thinning,
            min_token_threshold=request.min_token_threshold,
            if_add_node_summary=request.if_add_node_summary,
            summary_token_threshold=request.summary_token_threshold,
            if_add_doc_description=request.if_add_doc_description,
            if_add_node_text=request.if_add_node_text,
            if_add_node_id=request.if_add_node_id,
            ollama_base_url=request.ollama_base_url,
            ollama_timeout=request.ollama_timeout,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tree indexing failed: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    include_markdown: bool = Query(True, description="Include full markdown content"),
    db: Session = Depends(get_db),
):
    """Retrieve document metadata and optionally full markdown."""
    storage = DocumentStorageService(db)
    document = storage.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    response = {
        "id": document.id,
        "filename": getattr(document, "original_filename", None) or getattr(document, "filename", ""),
        "file_type": document.file_type,
        "total_pages": document.total_pages,
        "created_at": document.created_at.isoformat(),
    }
    if include_markdown:
        response["markdown"] = storage.get_document_markdown(document_id)
    return response


@router.get("/documents/{document_id}/markdown")
async def get_document_markdown(
    document_id: str,
    db: Session = Depends(get_db),
):
    """Get document markdown content only."""
    storage = DocumentStorageService(db)
    document = storage.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    markdown = storage.get_document_markdown(document_id)
    return PlainTextResponse(content=markdown)


@router.get("/documents/{document_id}/elements")
async def get_document_elements(
    document_id: str,
    label: Optional[str] = Query(None, description="Filter by label"),
    db: Session = Depends(get_db),
):
    """Get all layout elements for a document."""
    storage = DocumentStorageService(db)
    document = storage.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    elements = storage.get_document_elements(document_id, label_filter=label)
    result = []
    for elem in elements:
        page = db.query(Page).filter(Page.id == elem.page_id).first()
        result.append({
            "id": elem.id,
            "label": elem.label,
            "text_content": elem.text_content,
            "bbox": {
                "x1": elem.bbox_x1, "y1": elem.bbox_y1,
                "x2": elem.bbox_x2, "y2": elem.bbox_y2,
            },
            "bbox_normalized": {
                "x1": elem.bbox_norm_x1, "y1": elem.bbox_norm_y1,
                "x2": elem.bbox_norm_x2, "y2": elem.bbox_norm_y2,
            } if elem.bbox_norm_x1 is not None else None,
            "page_number": page.page_number if page else None,
            "page_id": elem.page_id,
            "sequence_order": elem.sequence_order,
            "has_crop_image": bool(elem.crop_image_base64),
        })
    return result


@router.get("/documents/{document_id}/tree")
async def get_tree_structure(
    document_id: str,
    db: Session = Depends(get_db),
):
    """Get tree index structure for a document."""
    tree_service = TreeIndexingService(db)
    tree = tree_service.get_tree_index(document_id)
    if not tree:
        raise HTTPException(
            status_code=404,
            detail="Tree index not found. Build it first with POST /build-index/{document_id}",
        )
    return tree


@router.get("/documents")
async def list_documents(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List all documents."""
    storage = DocumentStorageService(db)
    documents = storage.list_documents(limit=limit, offset=offset)
    return [
        {
            "id": doc.id,
            "filename": getattr(doc, "original_filename", None) or getattr(doc, "filename", ""),
            "file_type": doc.file_type,
            "total_pages": doc.total_pages,
            "created_at": doc.created_at.isoformat(),
        }
        for doc in documents
    ]
