"""
Repository pattern for data access.

Provides clean separation between data access and business logic.
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from data.db_models import Document, Page, LayoutElement, TreeIndex, TreeNode


class DocumentRepository:
    """Repository for Document operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, filename: str, file_type: str, total_pages: int) -> Document:
        """Create a new document."""
        document = Document(
            filename=filename,
            file_type=file_type,
            total_pages=total_pages
        )
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document
    
    def get_by_id(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.session.query(Document).filter(
            Document.id == document_id
        ).first()
    
    def list_all(self, limit: int = 50, offset: int = 0) -> List[Document]:
        """List documents with pagination."""
        return self.session.query(Document)\
            .order_by(Document.created_at.desc())\
            .limit(limit)\
            .offset(offset)\
            .all()
    
    def delete(self, document_id: str) -> bool:
        """Delete a document."""
        document = self.get_by_id(document_id)
        if document:
            self.session.delete(document)
            self.session.commit()
            return True
        return False


class PageRepository:
    """Repository for Page operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self,
        document_id: str,
        page_number: int,
        markdown_content: str,
        image_base64: str = "",
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ) -> Page:
        """Create a new page."""
        page = Page(
            document_id=document_id,
            page_number=page_number,
            markdown_content=markdown_content,
            image_base64=image_base64,
            image_width=image_width,
            image_height=image_height
        )
        self.session.add(page)
        self.session.commit()
        self.session.refresh(page)
        return page
    
    def get_by_document(self, document_id: str) -> List[Page]:
        """Get all pages for a document."""
        return self.session.query(Page)\
            .filter(Page.document_id == document_id)\
            .order_by(Page.page_number)\
            .all()


class LayoutElementRepository:
    """Repository for LayoutElement operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, page_id: str, element_data: dict) -> LayoutElement:
        """Create a layout element."""
        element = LayoutElement(
            page_id=page_id,
            **element_data
        )
        self.session.add(element)
        return element
    
    def get_by_document(
        self,
        document_id: str,
        label_filter: Optional[str] = None
    ) -> List[LayoutElement]:
        """Get layout elements for a document."""
        query = self.session.query(LayoutElement)\
            .join(Page)\
            .filter(Page.document_id == document_id)
        
        if label_filter:
            query = query.filter(LayoutElement.label == label_filter)
        
        return query.order_by(Page.page_number, LayoutElement.sequence_order).all()


class TreeIndexRepository:
    """Repository for TreeIndex operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self,
        document_id: str,
        tree_data: dict,
        config: dict
    ) -> TreeIndex:
        """Create a tree index."""
        tree_index = TreeIndex(
            document_id=document_id,
            tree_data=tree_data,
            config=config
        )
        self.session.add(tree_index)
        self.session.commit()
        self.session.refresh(tree_index)
        return tree_index
    
    def get_by_document(self, document_id: str) -> Optional[TreeIndex]:
        """Get tree index for a document."""
        return self.session.query(TreeIndex)\
            .filter(TreeIndex.document_id == document_id)\
            .order_by(TreeIndex.created_at.desc())\
            .first()
