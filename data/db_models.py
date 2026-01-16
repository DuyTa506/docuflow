"""
Database models for OCR document storage with metadata.

Stores documents, pages, layout elements (bounding boxes), and tree indices.
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


def generate_uuid():
    """Generate UUID string for primary keys."""
    return str(uuid.uuid4())


class Document(Base):
    """Top-level document metadata."""
    
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # 'pdf' or 'image'
    total_pages = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    tree_indices = relationship("TreeIndex", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, pages={self.total_pages})>"


class Page(Base):
    """Individual page content with markdown and image."""
    
    __tablename__ = 'pages'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey('documents.id'), nullable=False)
    page_number = Column(Integer, nullable=False)
    markdown_content = Column(Text, nullable=False)
    image_base64 = Column(Text)  # Optional: original page image
    image_width = Column(Integer)
    image_height = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="pages")
    layout_elements = relationship("LayoutElement", back_populates="page", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Page(id={self.id}, doc_id={self.document_id}, page_num={self.page_number})>"


class LayoutElement(Base):
    """Layout element with bounding box coordinates and metadata."""
    
    __tablename__ = 'layout_elements'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    page_id = Column(String, ForeignKey('pages.id'), nullable=False)
    
    # Element type (e.g., 'title', 'text', 'table', 'image', 'formula')
    label = Column(String, nullable=False)
    
    # Text content
    text_content = Column(Text)
    
    # Absolute pixel coordinates (scaled to image dimensions)
    bbox_x1 = Column(Integer, nullable=False)
    bbox_y1 = Column(Integer, nullable=False)
    bbox_x2 = Column(Integer, nullable=False)
    bbox_y2 = Column(Integer, nullable=False)
    
    # Normalized coordinates (0-999 range from DeepSeek OCR)
    bbox_norm_x1 = Column(Float)
    bbox_norm_y1 = Column(Float)
    bbox_norm_x2 = Column(Float)
    bbox_norm_y2 = Column(Float)
    
    # Cropped image (for 'image' label elements)
    crop_image_base64 = Column(Text)
    
    # Sequence order in document (for preserving reading order)
    sequence_order = Column(Integer)
    
    # Relationships
    page = relationship("Page", back_populates="layout_elements")
    
    def __repr__(self):
        return f"<LayoutElement(id={self.id}, label={self.label}, bbox=({self.bbox_x1},{self.bbox_y1})-({self.bbox_x2},{self.bbox_y2}))>"
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'page_id': self.page_id,
            'label': self.label,
            'text_content': self.text_content,
            'bbox': {
                'x1': self.bbox_x1,
                'y1': self.bbox_y1,
                'x2': self.bbox_x2,
                'y2': self.bbox_y2
            },
            'bbox_normalized': {
                'x1': self.bbox_norm_x1,
                'y1': self.bbox_norm_y1,
                'x2': self.bbox_norm_x2,
                'y2': self.bbox_norm_y2
            },
            'crop_image_base64': self.crop_image_base64,
            'sequence_order': self.sequence_order
        }


class TreeIndex(Base):
    """PageIndex tree structure storage."""
    
    __tablename__ = 'tree_indices'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey('documents.id'), nullable=False)
    
    # Full tree structure as JSON
    tree_data = Column(JSON, nullable=False)
    
    # PageIndex configuration used
    config = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="tree_indices")
    tree_nodes = relationship("TreeNode", back_populates="tree_index", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<TreeIndex(id={self.id}, doc_id={self.document_id})>"


class TreeNode(Base):
    """Individual tree node for efficient querying."""
    
    __tablename__ = 'tree_nodes'
    
    id = Column(String, primary_key=True, default=generate_uuid)
    tree_index_id = Column(String, ForeignKey('tree_indices.id'), nullable=False)
    
    # Node identification
    node_id = Column(String, nullable=False)  # PageIndex node ID
    node_type = Column(String)  # 'chapter', 'section', 'subsection', etc.
    
    # Content
    title = Column(String)
    summary = Column(Text)  # If node summary was generated
    
    # Hierarchy
    parent_node_id = Column(String)  # Reference to parent node's node_id
    
    # Page range
    page_start = Column(Integer)
    page_end = Column(Integer)
    
    # Token count (for thinning decisions)
    token_count = Column(Integer)
    
    # Relationships
    tree_index = relationship("TreeIndex", back_populates="tree_nodes")
    
    def __repr__(self):
        return f"<TreeNode(id={self.id}, node_id={self.node_id}, title={self.title})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'tree_index_id': self.tree_index_id,
            'node_id': self.node_id,
            'node_type': self.node_type,
            'title': self.title,
            'summary': self.summary,
            'parent_node_id': self.parent_node_id,
            'page_range': {
                'start': self.page_start,
                'end': self.page_end
            },
            'token_count': self.token_count
        }
