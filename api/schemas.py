"""
Pydantic schemas for API request/response validation.
"""
from typing import Optional, Dict
from pydantic import BaseModel


class TreeIndexRequest(BaseModel):
    """Request body for building tree index."""
    # PageIndex parameters
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
