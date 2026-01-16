"""
Document Storage Service

Handles CRUD operations for documents, pages, layout elements, and tree indices.
"""
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from PIL import Image
import base64
from io import BytesIO

from data.db_models import Document, Page, LayoutElement, TreeIndex, TreeNode
from .logic import ServicePageResult, LayoutElement as LogicLayoutElement


class DocumentStorageService:
    """Service for storing and retrieving OCR documents."""
    
    def __init__(self, session: Session):
        """
        Initialize storage service.
        
        Args:
            session: SQLAlchemy database session
        """
        self.session = session
    
    def create_document(
        self,
        filename: str,
        file_type: str,
        total_pages: int
    ) -> Document:
        """
        Create a new document entry.
        
        Args:
            filename: Original filename
            file_type: 'pdf' or 'image'
            total_pages: Total number of pages
        
        Returns:
            Created Document object
        """
        document = Document(
            filename=filename,
            file_type=file_type,
            total_pages=total_pages
        )
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document
    
    def save_page_result(
        self,
        document_id: str,
        page_result: ServicePageResult
    ) -> Page:
        """
        Save a page result from OCR processing.
        
        Args:
            document_id: ID of parent document
            page_result: ServicePageResult from OCR processing
        
        Returns:
            Created Page object
        """
        # Decode image to get dimensions
        img_width, img_height = None, None
        if page_result.image_base64:
            try:
                img_data = base64.b64decode(page_result.image_base64)
                img = Image.open(BytesIO(img_data))
                img_width, img_height = img.size
            except Exception as e:
                print(f"Warning: Could not decode image for dimension extraction: {e}")
        
        # Create page
        page = Page(
            document_id=document_id,
            page_number=page_result.page_num,
            markdown_content=page_result.markdown,
            image_base64=page_result.image_base64,
            image_width=img_width,
            image_height=img_height
        )
        self.session.add(page)
        self.session.flush()  # Get page ID before adding elements
        
        # Create layout elements
        if page_result.layout_elements:
            for idx, elem in enumerate(page_result.layout_elements):
                self._save_layout_element(page.id, elem, idx, img_width, img_height)
        
        self.session.commit()
        self.session.refresh(page)
        return page
    
    def _save_layout_element(
        self,
        page_id: str,
        element: Dict,
        sequence_order: int,
        img_width: Optional[int],
        img_height: Optional[int]
    ):
        """Save a layout element with bounding box."""
        # Handle both dict and object formats
        if isinstance(element, dict):
            label = element.get('label', '')
            text_content = element.get('text', '')
            x1, y1 = element.get('x1', 0), element.get('y1', 0)
            x2, y2 = element.get('x2', 0), element.get('y2', 0)
            crop_image = element.get('crop_image', '')
        else:
            # Assume it's a LayoutElement-like object
            label = getattr(element, 'label', '')
            text_content = getattr(element, 'text', '')
            x1 = getattr(element, 'x1', 0)
            y1 = getattr(element, 'y1', 0)
            x2 = getattr(element, 'x2', 0)
            y2 = getattr(element, 'y2', 0)
            crop_image = getattr(element, 'crop_image', '')
        
        # Calculate normalized coordinates (reverse of scaling)
        norm_x1 = norm_y1 = norm_x2 = norm_y2 = None
        if img_width and img_height:
            norm_x1 = (x1 / img_width) * 999.0
            norm_y1 = (y1 / img_height) * 999.0
            norm_x2 = (x2 / img_width) * 999.0
            norm_y2 = (y2 / img_height) * 999.0
        
        layout_elem = LayoutElement(
            page_id=page_id,
            label=label,
            text_content=text_content,
            bbox_x1=x1,
            bbox_y1=y1,
            bbox_x2=x2,
            bbox_y2=y2,
            bbox_norm_x1=norm_x1,
            bbox_norm_y1=norm_y1,
            bbox_norm_x2=norm_x2,
            bbox_norm_y2=norm_y2,
            crop_image_base64=crop_image,
            sequence_order=sequence_order
        )
        self.session.add(layout_elem)
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Document ID
        
        Returns:
            Document object or None
        """
        return self.session.query(Document).filter(
            Document.id == document_id
        ).first()
    
    def get_document_markdown(self, document_id: str) -> str:
        """
        Get complete markdown for a document (all pages concatenated).
        
        Args:
            document_id: Document ID
        
        Returns:
            Combined markdown content
        """
        pages = self.session.query(Page).filter(
            Page.document_id == document_id
        ).order_by(Page.page_number).all()
        
        markdown_parts = []
        for page in pages:
            markdown_parts.append(f"# Page {page.page_number}\n\n{page.markdown_content}")
        
        return "\n\n---\n\n".join(markdown_parts)
    
    def get_page_elements(
        self,
        page_id: str,
        label_filter: Optional[str] = None
    ) -> List[LayoutElement]:
        """
        Get layout elements for a page.
        
        Args:
            page_id: Page ID
            label_filter: Optional label to filter by (e.g., 'image', 'table')
        
        Returns:
            List of LayoutElement objects
        """
        query = self.session.query(LayoutElement).filter(
            LayoutElement.page_id == page_id
        )
        
        if label_filter:
            query = query.filter(LayoutElement.label == label_filter)
        
        return query.order_by(LayoutElement.sequence_order).all()
    
    def get_document_elements(
        self,
        document_id: str,
        label_filter: Optional[str] = None
    ) -> List[LayoutElement]:
        """
        Get all layout elements for a document.
        
        Args:
            document_id: Document ID
            label_filter: Optional label to filter by
        
        Returns:
            List of LayoutElement objects
        """
        query = self.session.query(LayoutElement).join(Page).filter(
            Page.document_id == document_id
        )
        
        if label_filter:
            query = query.filter(LayoutElement.label == label_filter)
        
        return query.order_by(Page.page_number, LayoutElement.sequence_order).all()
    
    def save_tree_index(
        self,
        document_id: str,
        tree_data: Dict,
        config: Optional[Dict] = None
    ) -> TreeIndex:
        """
        Save a tree index for a document.
        
        Args:
            document_id: Document ID
            tree_data: Tree structure as JSON/dict
            config: PageIndex configuration used
        
        Returns:
            Created TreeIndex object
        """
        tree_index = TreeIndex(
            document_id=document_id,
            tree_data=tree_data,
            config=config or {}
        )
        self.session.add(tree_index)
        self.session.flush()
        
        # Extract and save individual nodes for querying
        self._extract_tree_nodes(tree_index.id, tree_data)
        
        self.session.commit()
        self.session.refresh(tree_index)
        return tree_index
    
    def _extract_tree_nodes(
        self,
        tree_index_id: str,
        tree_data: Dict,
        parent_node_id: Optional[str] = None
    ):
        """Recursively extract tree nodes for storage."""
        # Extract node information
        node_id = tree_data.get('node_id', tree_data.get('id', ''))
        if not node_id:
            return
        
        node = TreeNode(
            tree_index_id=tree_index_id,
            node_id=node_id,
            node_type=tree_data.get('type', tree_data.get('node_type')),
            title=tree_data.get('title', tree_data.get('name')),
            summary=tree_data.get('summary', tree_data.get('node_summary')),
            parent_node_id=parent_node_id,
            page_start=tree_data.get('page_start', tree_data.get('start_page')),
            page_end=tree_data.get('page_end', tree_data.get('end_page')),
            token_count=tree_data.get('token_count', tree_data.get('tokens'))
        )
        self.session.add(node)
        
        # Process children recursively
        children = tree_data.get('children', tree_data.get('child_nodes', []))
        for child in children:
            self._extract_tree_nodes(tree_index_id, child, node_id)
    
    def get_tree_index(self, document_id: str) -> Optional[TreeIndex]:
        """
        Get the most recent tree index for a document.
        
        Args:
            document_id: Document ID
        
        Returns:
            TreeIndex object or None
        """
        return self.session.query(TreeIndex).filter(
            TreeIndex.document_id == document_id
        ).order_by(TreeIndex.created_at.desc()).first()
    
    def get_tree_nodes(
        self,
        tree_index_id: str,
        node_type: Optional[str] = None
    ) -> List[TreeNode]:
        """
        Get tree nodes for a tree index.
        
        Args:
            tree_index_id: Tree index ID
            node_type: Optional filter by node type
        
        Returns:
            List of TreeNode objects
        """
        query = self.session.query(TreeNode).filter(
            TreeNode.tree_index_id == tree_index_id
        )
        
        if node_type:
            query = query.filter(TreeNode.node_type == node_type)
        
        return query.all()
    
    def list_documents(self, limit: int = 50, offset: int = 0) -> List[Document]:
        """
        List all documents.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
        
        Returns:
            List of Document objects
        """
        return self.session.query(Document).order_by(
            Document.created_at.desc()
        ).limit(limit).offset(offset).all()
