"""
Workflow API for OCR-to-indexed-document pipeline.

Provides endpoints for:
- Document processing with OCR
- Tree index generation
- Document retrieval
- Downstream processing (summarization, translation)
"""
import os
import tempfile
import uuid
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from openai import AsyncOpenAI

from data.database import get_db, init_database
from data.db_models import Document, Page, LayoutElement
from .storage_service import DocumentStorageService
from .tree_indexing_service import TreeIndexingService
from .logic import process_page_api


# Configuration from environment
API_KEY = os.getenv("VLLM_API_KEY", "123")
SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1")


# Pydantic models for request/response
class TreeIndexRequest(BaseModel):
    """Request body for building tree index."""
    if_thinning: bool = True
    min_token_threshold: int = 5000
    if_add_node_summary: str = "yes"
    summary_token_threshold: int = 200
    model: str = "gpt-4o-2024-11-20"
    if_add_doc_description: str = "no"
    if_add_node_text: str = "no"
    if_add_node_id: str = "yes"
    llm_provider: str = "openai"
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 300
    # Spatial metadata parameters
    use_spatial_metadata: bool = True
    discover_implicit_sections: bool = True
    spatial_weights: Optional[dict] = None


class DocumentResponse(BaseModel):
    """Response for document metadata."""
    id: str
    filename: str
    file_type: str
    total_pages: int
    created_at: str
    markdown: Optional[str] = None


class LayoutElementResponse(BaseModel):
    """Response for layout element."""
    id: str
    label: str
    text_content: Optional[str]
    bbox: dict
    bbox_normalized: Optional[dict]
    page_number: int
    page_id: str


# Create FastAPI app
workflow_app = FastAPI(
    title="OCR Workflow API",
    description="End-to-end OCR processing with tree indexing and metadata storage",
    version="1.0.0"
)


# Initialize database on startup
@workflow_app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    init_database()
    print("✓ Workflow API initialized")


@workflow_app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    store_to_db: bool = Query(True, description="Store results to database"),
    db: Session = Depends(get_db)
):
    """
    Process PDF/image through OCR and optionally store to database.
    
    Args:
        file: PDF or image file to process
        store_to_db: Whether to persist to database
        db: Database session
    
    Returns:
        Document metadata with document_id
    """
    # Save uploaded file to temp location
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=os.path.splitext(file.filename)[1]
    )
    
    try:
        # Write uploaded file
        content = await file.read()
        with os.fdopen(temp_fd, 'wb') as f:
            f.write(content)
        
        # Determine file type
        file_type = 'pdf' if file.filename.lower().endswith('.pdf') else 'image'
        
        # Count pages
        if file_type == 'pdf':
            from PyPDF2 import PdfReader
            reader = PdfReader(temp_path)
            num_pages = len(reader.pages)
        else:
            num_pages = 1
        
        # Create document in database if storing
        storage = DocumentStorageService(db)
        document = None
        if store_to_db:
            document = storage.create_document(
                filename=file.filename,
                file_type=file_type,
                total_pages=num_pages
            )
        
        # Process with OCR
        client = AsyncOpenAI(api_key=API_KEY, base_url=SERVER_URL)
        
        element_count = 0
        for page_num in range(1, num_pages + 1):
            # Process page
            page_result = None
            async for event in process_page_api(
                client=client,
                pdf_path=temp_path,
                page_num=page_num,
                stream_enabled=False
            ):
                if event.get("type") == "result":
                    page_result = event["result"]
            
            # Save to database
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
            "stored_to_db": store_to_db
        }
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@workflow_app.post("/build-index/{document_id}")
async def build_index(
    document_id: str,
    request: TreeIndexRequest,
    db: Session = Depends(get_db)
):
    """
    Build PageIndex tree structure from stored document.
    
    Supports both standard markdown-based indexing and enhanced spatial-aware indexing.
    
    Args:
        document_id: ID of document to index
        request: Tree index configuration
        db: Database session
    
    Returns:
        Tree index metadata
    """
    tree_service = TreeIndexingService(
        session=db,
        llm_provider=request.llm_provider,
        model=request.model
    )
    
    try:
        # Use enhanced tree builder with spatial metadata
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
            ollama_timeout=request.ollama_timeout
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tree indexing failed: {str(e)}")


@workflow_app.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    include_markdown: bool = Query(True, description="Include full markdown content"),
    db: Session = Depends(get_db)
):
    """
    Retrieve document metadata and optionally full markdown.
    
    Args:
        document_id: Document ID
        include_markdown: Whether to include full markdown
        db: Database session
    
    Returns:
        Document metadata
    """
    storage = DocumentStorageService(db)
    document = storage.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    response = {
        "id": document.id,
        "filename": document.filename,
        "file_type": document.file_type,
        "total_pages": document.total_pages,
        "created_at": document.created_at.isoformat()
    }
    
    if include_markdown:
        response["markdown"] = storage.get_document_markdown(document_id)
    
    return response


@workflow_app.get("/documents/{document_id}/markdown")
async def get_document_markdown(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get document markdown content only.
    
    Args:
        document_id: Document ID
        db: Database session
    
    Returns:
        Plain text markdown
    """
    storage = DocumentStorageService(db)
    document = storage.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    markdown = storage.get_document_markdown(document_id)
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=markdown)


@workflow_app.get("/documents/{document_id}/elements")
async def get_document_elements(
    document_id: str,
    label: Optional[str] = Query(None, description="Filter by label (e.g., 'image', 'table')"),
    db: Session = Depends(get_db)
):
    """
    Get all layout elements for a document.
    
    Args:
        document_id: Document ID
        label: Optional filter by element label
        db: Database session
    
    Returns:
        List of layout elements with bounding boxes
    """
    storage = DocumentStorageService(db)
    
    # Verify document exists
    document = storage.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    elements = storage.get_document_elements(document_id, label_filter=label)
    
    # Convert to response format
    result = []
    for elem in elements:
        # Get page number from page relationship
        page = db.query(Page).filter(Page.id == elem.page_id).first()
        
        result.append({
            "id": elem.id,
            "label": elem.label,
            "text_content": elem.text_content,
            "bbox": {
                "x1": elem.bbox_x1,
                "y1": elem.bbox_y1,
                "x2": elem.bbox_x2,
                "y2": elem.bbox_y2
            },
            "bbox_normalized": {
                "x1": elem.bbox_norm_x1,
                "y1": elem.bbox_norm_y1,
                "x2": elem.bbox_norm_x2,
                "y2": elem.bbox_norm_y2
            } if elem.bbox_norm_x1 is not None else None,
            "page_number": page.page_number if page else None,
            "page_id": elem.page_id,
            "sequence_order": elem.sequence_order,
            "has_crop_image": bool(elem.crop_image_base64)
        })
    
    return result


@workflow_app.get("/documents/{document_id}/tree")
async def get_tree_structure(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get tree index structure for a document.
    
    Args:
        document_id: Document ID
        db: Database session
    
    Returns:
        Tree structure with metadata
    """
    tree_service = TreeIndexingService(db)
    tree = tree_service.get_tree_index(document_id)
    
    if not tree:
        raise HTTPException(
            status_code=404,
            detail="Tree index not found. Build it first with POST /build-index/{document_id}"
        )
    
    return tree


@workflow_app.get("/documents")
async def list_documents(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    List all documents.
    
    Args:
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        db: Database session
    
    Returns:
        List of documents with metadata
    """
    storage = DocumentStorageService(db)
    documents = storage.list_documents(limit=limit, offset=offset)
    
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "total_pages": doc.total_pages,
            "created_at": doc.created_at.isoformat()
        }
        for doc in documents
    ]


@workflow_app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "OCR Workflow API",
        "version": "1.0.0",
        "endpoints": {
            "process_document": "POST /process-document",
            "build_index": "POST /build-index/{document_id}",
            "get_document": "GET /documents/{document_id}",
            "get_elements": "GET /documents/{document_id}/elements",
            "get_tree": "GET /documents/{document_id}/tree",
            "list_documents": "GET /documents"
        }
    }


# Export app for uvicorn
app = workflow_app
