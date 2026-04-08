"""
Pydantic schemas for API request/response validation.

Covers: auth, documents, tasks, translations, summaries, keywords,
research directions, main content, search.

This is the single source of truth for ALL Pydantic I/O schemas.
Duplicate definitions in serving/workflow_api.py have been removed.
"""
from typing import Literal, Optional, Dict, List, Any
from pydantic import BaseModel, Field


# ── Existing schemas (preserved) ────────────────────────────────────

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
    use_spatial_metadata: bool = True
    discover_implicit_sections: bool = True
    spatial_weights: Optional[Dict[str, float]] = None


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


# ── Auth schemas ────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """
    Self-registration request.

    group: TEACHER (academic staff) or LIBRARY (library staff)
    role:  MEMBER (standard) — ADMIN accounts are created out-of-band only.
    All self-registered accounts start as PENDING_APPROVAL until an admin activates them.
    """
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    email: Optional[str] = None
    group: Literal["TEACHER", "LIBRARY"] = Field(
        default="TEACHER",
        description="User group: TEACHER (academic) or LIBRARY (library staff).",
    )
    role: Literal["MEMBER"] = Field(
        default="MEMBER",
        description="Permission level. Only MEMBER is allowed for self-registration.",
    )


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    username: str
    full_name: Optional[str]
    email: Optional[str] = None
    group: str
    role: str
    status: str
    created_at: Optional[str] = None


# ── Task schemas ────────────────────────────────────────────────────

class TaskResponse(BaseModel):
    task_id: str
    document_id: Optional[str] = None
    task_type: str
    status: str
    progress: int = 0
    message: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TaskSubmittedResponse(BaseModel):
    """Returned when a background task is created."""
    task_id: str
    status: str = "PENDING"
    message: str = "Task submitted"


# ── Document V2 schemas ─────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: str
    title: str
    format: Optional[str] = None
    total_pages: int
    processing_status: str


class DocumentDetailResponse(BaseModel):
    id: str
    title: str
    original_filename: Optional[str] = None
    source_language: Optional[str] = None
    format: Optional[str] = None
    file_type: Optional[str] = None
    total_pages: int
    processing_status: str
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DocumentTextResponse(BaseModel):
    document_id: str
    ocr_content: Optional[str] = None
    normalized_content: Optional[str] = None


class PageResponse(BaseModel):
    id: str
    page_number: int
    markdown_content: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None


# ── Translation schemas ─────────────────────────────────────────────

class TranslationRequest(BaseModel):
    target_language: str = "vi"
    domain: Literal["general", "military", "education", "science"] = "general"


class TranslationResponse(BaseModel):
    id: str
    document_id: str
    target_language: str
    translated_content: Optional[str] = None
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TranslationEditRequest(BaseModel):
    translated_content: str


# ── Summary schemas ──────────────────────────────────────────────────

class SummaryRequest(BaseModel):
    summary_type: str = "short"  # short | detailed | hierarchical


class SummaryResponse(BaseModel):
    id: str
    document_id: str
    summary_type: str
    content: Optional[str] = None
    created_at: Optional[str] = None


# ── Main Content schemas ─────────────────────────────────────────────

class MainContentResponse(BaseModel):
    id: str
    document_id: str
    details: Optional[Dict] = None
    created_at: Optional[str] = None


# ── Keyword schemas ──────────────────────────────────────────────────

class KeywordWithWeight(BaseModel):
    keyword: str
    weight: float


class KeywordsRequest(BaseModel):
    max_keywords: int = 20


class KeywordsResponse(BaseModel):
    document_id: str
    keywords: List[KeywordWithWeight]


# ── Research Direction schemas ───────────────────────────────────────

class ResearchDirectionItem(BaseModel):
    direction_name: str
    confidence: float
    reasoning: Optional[str] = None
    is_predefined: bool = True


class ResearchDirectionsResponse(BaseModel):
    document_id: str
    directions: List[ResearchDirectionItem]


class CatalogDirectionRequest(BaseModel):
    direction_name: str


class CatalogDirectionResponse(BaseModel):
    id: str
    direction_name: str
    is_predefined: bool
    created_at: Optional[str] = None


# ── Search schemas ───────────────────────────────────────────────────

class SearchResultItem(BaseModel):
    document_id: str
    title: str
    snippet: Optional[str] = None
    match_field: str  # title, content, keywords, translations


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    total: int
    query: str


# ── List-endpoint schemas (Phase 4 additions) ────────────────────────

class DocumentListItem(BaseModel):
    """Single row returned by GET /api/v2/documents."""
    id: str
    title: str
    original_filename: Optional[str] = None
    format: Optional[str] = None
    total_pages: int
    processing_status: str
    source_language: Optional[str] = None
    created_at: Optional[str] = None
    task_summary: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Latest status of each pipeline task for this document. "
            "Keys: EXTRACT, TRANSLATE, SUMMARIZE, KEYWORDS, RESEARCH_DIRECTIONS, MAIN_CONTENT. "
            "Values: PENDING | RUNNING | COMPLETED | FAILED. Absent key = never started."
        ),
    )


class PageListItem(BaseModel):
    """Single row returned by GET /api/v2/documents/{id}/pages."""
    id: str
    page_number: int
    markdown_content: str
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class ElementListItem(BaseModel):
    """Single row returned by GET /api/v2/documents/{id}/elements."""
    id: str
    label: str
    text_content: Optional[str] = None
    bbox: dict
    bbox_normalized: Optional[dict] = None
    page_number: Optional[int] = None
    page_id: str
    sequence_order: Optional[int] = None
    has_crop_image: bool = False


class SummaryListItem(BaseModel):
    """Single row returned by GET /api/v2/documents/{id}/summaries."""
    id: str
    document_id: str
    summary_type: str
    content: Optional[str] = None
    created_at: Optional[str] = None


class DigestResponse(BaseModel):
    """Response for POST /api/v2/documents/{id}/digest."""
    document_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    main_content: Optional[dict] = None
    keywords: Optional[list] = None
    research_directions: Optional[list] = None
    missing: Optional[list] = None


class TranslationListItem(BaseModel):
    """Single row returned by GET /api/v2/documents/{id}/translations."""
    id: str
    document_id: str
    target_language: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ResearchListItem(BaseModel):
    """Single item in research-directions response."""
    direction_name: str
    confidence: float
    reasoning: Optional[str] = None
    is_predefined: bool = True


class ResearchDirectionsResponse(BaseModel):
    """Response for GET /api/v2/documents/{id}/research-directions."""
    document_id: str
    directions: List[ResearchListItem]


class CatalogListItem(BaseModel):
    """Single row returned by GET /api/v2/research-directions/catalog."""
    id: str
    direction_name: str
    is_predefined: bool
    created_at: Optional[str] = None


class TaskListItem(BaseModel):
    """Single row returned by GET /api/v2/tasks."""
    task_id: str
    document_id: Optional[str] = None
    task_type: str
    status: str
    progress: int = 0
    message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
