"""
Unit tests for data.db_models module.
"""
import pytest
from datetime import datetime
from data.db_models import Document, Page, LayoutElement, TreeIndex, TreeNode


class TestDocument:
    """Tests for Document model."""
    
    def test_create_document(self, test_db_session):
        """Test creating a document."""
        doc = Document(
            filename="test.pdf",
            file_type="pdf",
            total_pages=5
        )
        
        test_db_session.add(doc)
        test_db_session.commit()
        
        assert doc.id is not None
        assert doc.filename == "test.pdf"
        assert doc.file_type == "pdf"
        assert doc.total_pages == 5
    
    def test_document_timestamps(self, test_db_session):
        """Test created_at and updated_at are set."""
        doc = Document(
            filename="test.pdf",
            file_type="pdf",
            total_pages=1
        )
        
        test_db_session.add(doc)
        test_db_session.commit()
        
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.updated_at, datetime)
    
    def test_document_relationships(self, test_db_session):
        """Test pages relationship."""
        doc = Document(filename="test.pdf", file_type="pdf", total_pages=2)
        test_db_session.add(doc)
        test_db_session.flush()
        
        page1 = Page(
            document_id=doc.id,
            page_number=1,
            markdown_content="# Page 1"
        )
        test_db_session.add(page1)
        test_db_session.commit()
        
        # Refresh to load relationships
        test_db_session.refresh(doc)
        
        assert len(doc.pages) == 1
        assert doc.pages[0].page_number == 1


class TestPage:
    """Tests for Page model."""
    
    def test_create_page(self, test_db_session):
        """Test creating a page."""
        doc = Document(filename="test.pdf", file_type="pdf", total_pages=1)
        test_db_session.add(doc)
        test_db_session.flush()
        
        page = Page(
            document_id=doc.id,
            page_number=1,
            markdown_content="# Test"
        )
        
        test_db_session.add(page)
        test_db_session.commit()
        
        assert page.id is not None
        assert page.document_id == doc.id
        assert page.page_number == 1
    
    def test_page_with_image_dimensions(self, test_db_session):
        """Test storing image dimensions."""
        doc = Document(filename="test.pdf", file_type="pdf", total_pages=1)
        test_db_session.add(doc)
        test_db_session.flush()
        
        page = Page(
            document_id=doc.id,
            page_number=1,
            markdown_content="test",
            image_width=800,
            image_height=600
        )
        
        test_db_session.add(page)
        test_db_session.commit()
        
        assert page.image_width == 800
        assert page.image_height == 600


class TestLayoutElement:
    """Tests for LayoutElement model."""
    
    def test_create_layout_element(self, test_db_session):
        """Test creating a layout element."""
        # Create document and page first
        doc = Document(filename="test.pdf", file_type="pdf", total_pages=1)
        test_db_session.add(doc)
        test_db_session.flush()
        
        page = Page(
            document_id=doc.id,
            page_number=1,
            markdown_content="test"
        )
        test_db_session.add(page)
        test_db_session.flush()
        
        # Create layout element
        element = LayoutElement(
            page_id=page.id,
            label="title",
            text_content="Test Title",
            bbox_x1=10,
            bbox_y1=20,
            bbox_x2=100,
            bbox_y2=50
        )
        
        test_db_session.add(element)
        test_db_session.commit()
        
        assert element.id is not None
        assert element.label == "title"
        assert element.bbox_x1 == 10
    
    def test_normalized_bbox_storage(self, test_db_session):
        """Test storing normalized bounding box."""
        doc = Document(filename="test.pdf", file_type="pdf", total_pages=1)
        test_db_session.add(doc)
        test_db_session.flush()
        
        page = Page(document_id=doc.id, page_number=1, markdown_content="test")
        test_db_session.add(page)
        test_db_session.flush()
        
        element = LayoutElement(
            page_id=page.id,
            label="text",
            bbox_x1=100,
            bbox_y1=200,
            bbox_x2=500,
            bbox_y2=600,
            bbox_norm_x1=100.0,
            bbox_norm_y1=200.0,
            bbox_norm_x2=500.0,
            bbox_norm_y2=600.0
        )
        
        test_db_session.add(element)
        test_db_session.commit()
        
        assert element.bbox_norm_x1 == 100.0
        assert element.bbox_norm_y2 == 600.0


class TestTreeIndex:
    """Tests for TreeIndex model."""
    
    def test_create_tree_index(self, test_db_session):
        """Test creating a tree index."""
        doc = Document(filename="test.pdf", file_type="pdf", total_pages=1)
        test_db_session.add(doc)
        test_db_session.flush()
        
        tree_data = {'title': 'Test', 'children': []}
        config = {'llm_provider': 'openai'}
        
        tree_index = TreeIndex(
            document_id=doc.id,
            tree_data=tree_data,
            config=config
        )
        
        test_db_session.add(tree_index)
        test_db_session.commit()
        
        assert tree_index.id is not None
        assert tree_index.tree_data == tree_data
        assert tree_index.config == config
    
    def test_tree_index_timestamp(self, test_db_session):
        """Test created_at is set."""
        doc = Document(filename="test.pdf", file_type="pdf", total_pages=1)
        test_db_session.add(doc)
        test_db_session.flush()
        
        tree_index = TreeIndex(
            document_id=doc.id,
            tree_data={},
            config={}
        )
        
        test_db_session.add(tree_index)
        test_db_session.commit()
        
        assert isinstance(tree_index.created_at, datetime)


class TestTreeNode:
    """Tests for TreeNode model."""
    
    def test_create_tree_node(self, test_db_session):
        """Test creating a tree node."""
        doc = Document(filename="test.pdf", file_type="pdf", total_pages=1)
        test_db_session.add(doc)
        test_db_session.flush()
        
        tree_index = TreeIndex(
            document_id=doc.id,
            tree_data={},
            config={}
        )
        test_db_session.add(tree_index)
        test_db_session.flush()
        
        node = TreeNode(
            tree_index_id=tree_index.id,
            node_id="node_1",
            title="Test Node",
            node_type="section"
        )
        
        test_db_session.add(node)
        test_db_session.commit()
        
        assert node.id is not None
        assert node.node_id == "node_1"
        assert node.title == "Test Node"
