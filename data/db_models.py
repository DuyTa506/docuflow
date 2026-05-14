"""
Database models for DocuFlow library management system.

Full schema with 16 tables for: documents, pages, layout elements, tree indices,
users, translations, summaries, keywords, research directions, tasks, etc.
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, ForeignKey, JSON, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


def generate_uuid():
    """Generate UUID string for primary keys."""
    return str(uuid.uuid4())


# ─── ID Sequence table (for prefixed IDs) ──────────────────────────

class IdSequence(Base):
    """Tracks current sequence value for each prefixed-ID table."""

    __tablename__ = "id_sequences"

    table_name = Column(String, primary_key=True)  # e.g. "users"
    prefix = Column(String, nullable=False)          # e.g. "USR"
    current_value = Column(Integer, nullable=False, default=0)

    def __repr__(self):
        return f"<IdSequence(table={self.table_name}, prefix={self.prefix}, current={self.current_value})>"


# ─── Users ──────────────────────────────────────────────────────────

class User(Base):
    """
    User accounts with group + role access control.

    Groups (user category):
      TEACHER  — academic staff / faculty
      LIBRARY  — library staff

    Roles (permission level):
      MEMBER   — standard access; can upload and manage their own documents
      ADMIN    — elevated access; can approve/deactivate accounts, see all documents

    Status lifecycle:
      PENDING_APPROVAL → (admin approves) → ACTIVE
      ACTIVE           → (admin deactivates) → DEACTIVATED
      MEMBER accounts self-register and start PENDING_APPROVAL.
      ADMIN accounts are created out-of-band.
    """

    __tablename__ = "users"

    id = Column(String, primary_key=True)               # USR_001
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    email = Column(String, unique=True, nullable=True)
    group = Column(String, nullable=False, default="TEACHER")   # TEACHER, LIBRARY
    role = Column(String, nullable=False, default="MEMBER")     # MEMBER, ADMIN
    status = Column(String, nullable=False, default="PENDING_APPROVAL")  # ACTIVE, PENDING_APPROVAL, DEACTIVATED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, group={self.group}, role={self.role})>"


# ─── Documents ──────────────────────────────────────────────────────

class Document(Base):
    """Document metadata."""

    __tablename__ = "documents"

    id = Column(String, primary_key=True)               # DOC_001
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    title = Column(String, nullable=False)
    original_filename = Column(String, nullable=True)
    source_language = Column(String, nullable=True, default="en")
    format = Column(String, nullable=True)               # pdf, image, docx
    file_path = Column(String, nullable=True)             # path on disk
    file_type = Column(String, nullable=True)             # backward compat
    total_pages = Column(Integer, nullable=True, default=0)
    processing_status = Column(String, nullable=False, default="INIT")
    # Statuses: INIT, EXTRACT_IN_PROGRESS, EXTRACTED, FAILED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="documents")
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    tree_indices = relationship("TreeIndex", back_populates="document", cascade="all, delete-orphan")
    digitized_texts = relationship("DigitizedText", back_populates="document", cascade="all, delete-orphan")
    translations = relationship("Translation", back_populates="document", cascade="all, delete-orphan")
    summaries = relationship("Summary", back_populates="document", cascade="all, delete-orphan")
    main_contents = relationship("MainContent", back_populates="document", cascade="all, delete-orphan")
    document_keywords = relationship("DocumentKeyword", back_populates="document", cascade="all, delete-orphan")
    document_research_directions = relationship("DocumentResearchDirection", back_populates="document", cascade="all, delete-orphan")
    keyword_extractions = relationship("KeywordExtraction", back_populates="document", cascade="all, delete-orphan")
    research_extractions = relationship("ResearchExtraction", back_populates="document", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="document", cascade="all, delete-orphan")

    @property
    def filename(self):
        """Backward compatibility: old code accesses doc.filename."""
        return self.original_filename or self.title

    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title}, pages={self.total_pages})>"


# ─── Pages ──────────────────────────────────────────────────────────

class Page(Base):
    """Individual page content with markdown and image."""

    __tablename__ = "pages"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    markdown_content = Column(Text, nullable=False)
    image_base64 = Column(Text)
    image_width = Column(Integer)
    image_height = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="pages")
    layout_elements = relationship("LayoutElement", back_populates="page", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Page(id={self.id}, doc_id={self.document_id}, page_num={self.page_number})>"


# ─── Layout Elements ────────────────────────────────────────────────

class LayoutElement(Base):
    """Layout element with bounding box coordinates and metadata."""

    __tablename__ = "layout_elements"

    id = Column(String, primary_key=True, default=generate_uuid)
    page_id = Column(String, ForeignKey("pages.id"), nullable=False)

    label = Column(String, nullable=False)
    text_content = Column(Text)

    # Absolute pixel coordinates
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)

    # Normalized coordinates (0-999)
    bbox_norm_x1 = Column(Float)
    bbox_norm_y1 = Column(Float)
    bbox_norm_x2 = Column(Float)
    bbox_norm_y2 = Column(Float)

    crop_image_base64 = Column(Text)
    sequence_order = Column(Integer)

    # Relationships
    page = relationship("Page", back_populates="layout_elements")

    def __repr__(self):
        return f"<LayoutElement(id={self.id}, label={self.label}, bbox=({self.bbox_x1},{self.bbox_y1})-({self.bbox_x2},{self.bbox_y2}))>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "page_id": self.page_id,
            "label": self.label,
            "text_content": self.text_content,
            "bbox": {
                "x1": self.bbox_x1,
                "y1": self.bbox_y1,
                "x2": self.bbox_x2,
                "y2": self.bbox_y2,
            },
            "bbox_normalized": {
                "x1": self.bbox_norm_x1,
                "y1": self.bbox_norm_y1,
                "x2": self.bbox_norm_x2,
                "y2": self.bbox_norm_y2,
            },
            "crop_image_base64": self.crop_image_base64,
            "sequence_order": self.sequence_order,
        }


# ─── Digitized Texts ────────────────────────────────────────────────

class DigitizedText(Base):
    """Aggregated OCR output + normalized text for a document."""

    __tablename__ = "digitized_texts"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    ocr_content = Column(Text, nullable=True)
    normalized_content = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="digitized_texts")

    def __repr__(self):
        return f"<DigitizedText(id={self.id}, doc_id={self.document_id})>"


# ─── Translations ────────────────────────────────────────────────────

class Translation(Base):
    """Multi-language translation of a document."""

    __tablename__ = "translations"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    target_language = Column(String, nullable=False, default="vi")
    translated_content = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="PENDING")
    # Unified job statuses: PENDING, IN_PROGRESS, COMPLETED, FAILED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="translations")

    def __repr__(self):
        return f"<Translation(id={self.id}, doc_id={self.document_id}, lang={self.target_language})>"


# ─── Summaries ───────────────────────────────────────────────────────

class Summary(Base):
    """Document summaries (short / detailed / hierarchical)."""

    __tablename__ = "summaries"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    summary_type = Column(String, nullable=False, default="short")  # short, detailed, hierarchical
    content = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="PENDING")
    # Unified job statuses: PENDING, IN_PROGRESS, COMPLETED, FAILED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="summaries")

    def __repr__(self):
        return f"<Summary(id={self.id}, doc_id={self.document_id}, type={self.summary_type}, status={self.status})>"


# ─── Main Contents ───────────────────────────────────────────────────

class MainContent(Base):
    """Structured key-points / main content extraction."""

    __tablename__ = "main_contents"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    details = Column(JSON, nullable=True)
    # Expected JSON shape:
    # {"key_points": [...], "methods": [...], "results": [...], "conclusions": [...]}
    status = Column(String, nullable=False, default="PENDING")
    # Unified job statuses: PENDING, IN_PROGRESS, COMPLETED, FAILED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="main_contents")

    def __repr__(self):
        return f"<MainContent(id={self.id}, doc_id={self.document_id}, status={self.status})>"


# ─── Keywords ────────────────────────────────────────────────────────

class Keyword(Base):
    """Global keyword dictionary."""

    __tablename__ = "keywords"

    id = Column(String, primary_key=True, default=generate_uuid)
    keyword_name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    document_keywords = relationship("DocumentKeyword", back_populates="keyword", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Keyword(id={self.id}, name={self.keyword_name})>"


class DocumentKeyword(Base):
    """Many-to-many: document ↔ keyword with weight (current keywords for the document)."""

    __tablename__ = "document_keywords"

    document_id = Column(String, ForeignKey("documents.id"), primary_key=True)
    keyword_id = Column(String, ForeignKey("keywords.id"), primary_key=True)
    weight = Column(Float, nullable=False, default=1.0)

    document = relationship("Document", back_populates="document_keywords")
    keyword = relationship("Keyword", back_populates="document_keywords")

    def __repr__(self):
        return f"<DocumentKeyword(doc={self.document_id}, kw={self.keyword_id}, w={self.weight})>"


class KeywordExtraction(Base):
    """Job tracker for a single keyword-extraction run on a document.

    The actual current keywords live in `document_keywords` (replaced on each run).
    This table preserves the history of extraction jobs and their statuses.
    """

    __tablename__ = "keyword_extractions"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    status = Column(String, nullable=False, default="PENDING")
    # Unified job statuses: PENDING, IN_PROGRESS, COMPLETED, FAILED
    max_keywords = Column(Integer, nullable=False, default=20)
    total_keywords = Column(Integer, nullable=True)  # populated on COMPLETED
    error = Column(Text, nullable=True)              # populated on FAILED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="keyword_extractions")

    def __repr__(self):
        return f"<KeywordExtraction(id={self.id}, doc={self.document_id}, status={self.status})>"


# ─── Research Directions ─────────────────────────────────────────────

class ResearchDirection(Base):
    """Catalog of research directions (predefined + discovered)."""

    __tablename__ = "research_directions"

    id = Column(String, primary_key=True, default=generate_uuid)
    direction_name = Column(String, unique=True, nullable=False)
    is_predefined = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document_research_directions = relationship(
        "DocumentResearchDirection", back_populates="research_direction", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<ResearchDirection(id={self.id}, name={self.direction_name})>"


class DocumentResearchDirection(Base):
    """Many-to-many: document ↔ research direction with confidence (current state)."""

    __tablename__ = "document_research_directions"

    document_id = Column(String, ForeignKey("documents.id"), primary_key=True)
    direction_id = Column(String, ForeignKey("research_directions.id"), primary_key=True)
    confidence = Column(Float, nullable=False, default=0.0)
    reasoning = Column(Text, nullable=True)

    document = relationship("Document", back_populates="document_research_directions")
    research_direction = relationship("ResearchDirection", back_populates="document_research_directions")

    def __repr__(self):
        return f"<DocumentResearchDirection(doc={self.document_id}, dir={self.direction_id})>"


class ResearchExtraction(Base):
    """Job tracker for a single research-direction extraction run.

    The actual current directions live in `document_research_directions` (replaced on each run).
    This table preserves the history of extraction jobs and their statuses.
    """

    __tablename__ = "research_extractions"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    status = Column(String, nullable=False, default="PENDING")
    # Unified job statuses: PENDING, IN_PROGRESS, COMPLETED, FAILED
    total_directions = Column(Integer, nullable=True)  # populated on COMPLETED
    error = Column(Text, nullable=True)                # populated on FAILED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="research_extractions")

    def __repr__(self):
        return f"<ResearchExtraction(id={self.id}, doc={self.document_id}, status={self.status})>"


# ─── Tree Indices ────────────────────────────────────────────────────

class TreeIndex(Base):
    """PageIndex tree structure storage."""

    __tablename__ = "tree_indices"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    tree_data = Column(JSON, nullable=False)
    config = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="tree_indices")
    tree_nodes = relationship("TreeNode", back_populates="tree_index", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TreeIndex(id={self.id}, doc_id={self.document_id})>"


class TreeNode(Base):
    """Individual tree node for efficient querying."""

    __tablename__ = "tree_nodes"

    id = Column(String, primary_key=True, default=generate_uuid)
    tree_index_id = Column(String, ForeignKey("tree_indices.id"), nullable=False)
    node_id = Column(String, nullable=False)
    node_type = Column(String)
    title = Column(String)
    summary = Column(Text)
    parent_node_id = Column(String)
    page_start = Column(Integer)
    page_end = Column(Integer)
    token_count = Column(Integer)

    tree_index = relationship("TreeIndex", back_populates="tree_nodes")

    def __repr__(self):
        return f"<TreeNode(id={self.id}, node_id={self.node_id}, title={self.title})>"

    def to_dict(self):
        return {
            "id": self.id,
            "tree_index_id": self.tree_index_id,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "title": self.title,
            "summary": self.summary,
            "parent_node_id": self.parent_node_id,
            "page_range": {"start": self.page_start, "end": self.page_end},
            "token_count": self.token_count,
        }


# ─── Background Tasks ────────────────────────────────────────────────

class Task(Base):
    """Background task tracking with DB-backed status."""

    __tablename__ = "tasks"

    id = Column(String, primary_key=True)               # TASK_001
    document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    task_type = Column(String, nullable=False)
    # Types: OCR, NORMALIZE, TRANSLATE, SUMMARIZE, KEYWORDS, RESEARCH_DIRECTIONS, MAIN_CONTENT, BUILD_TREE
    status = Column(String, nullable=False, default="PENDING")
    # Statuses: PENDING, RUNNING, COMPLETED, FAILED
    progress = Column(Integer, nullable=False, default=0)       # 0–100
    message = Column(String, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="tasks")

    def __repr__(self):
        return f"<Task(id={self.id}, type={self.task_type}, status={self.status})>"
