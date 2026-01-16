"""
Unit tests for core.models module.
"""
import pytest
from core.models import ServicePageResult, LayoutElement, BoundingBox


class TestServicePageResult:
    """Tests for ServicePageResult dataclass."""
    
    def test_initialization(self):
        """Test creating ServicePageResult with required fields."""
        result = ServicePageResult(
            page_num=1,
            markdown="# Test"
        )
        
        assert result.page_num == 1
        assert result.markdown == "# Test"
        assert result.input_tokens == 0
        assert result.output_tokens == 0
    
    def test_default_values(self):
        """Test default values for optional fields."""
        result = ServicePageResult(page_num=1, markdown="test")
        
        assert result.image_base64 == ""
        assert result.annotated_image_base64 == ""
        assert isinstance(result.layout_elements, list)
        assert len(result.layout_elements) == 0
        assert isinstance(result.crops_base64, list)
    
    def test_layout_elements_default_list(self):
        """Test layout_elements defaults to empty list."""
        result = ServicePageResult(page_num=1, markdown="test")
        
        # Should be mutable and independent
        result.layout_elements.append({'test': 'data'})
        assert len(result.layout_elements) == 1
        
        # New instance should have empty list
        result2 = ServicePageResult(page_num=2, markdown="test2")
        assert len(result2.layout_elements) == 0
    
    def test_with_all_fields(self):
        """Test creating with all fields populated."""
        elements = [{'label': 'title', 'text': 'Test'}]
        crops = ['base64data']
        
        result = ServicePageResult(
            page_num=5,
            markdown="# Title",
            input_tokens=100,
            output_tokens=200,
            image_base64="img_data",
            annotated_image_base64="annotated_data",
            layout_elements=elements,
            crops_base64=crops
        )
        
        assert result.page_num == 5
        assert result.input_tokens == 100
        assert result.output_tokens == 200
        assert len(result.layout_elements) == 1
        assert len(result.crops_base64) == 1


class TestLayoutElement:
    """Tests for LayoutElement dataclass."""
    
    def test_initialization(self):
        """Test creating LayoutElement."""
        element = LayoutElement(
            label="title",
            x1=10,
            y1=20,
            x2=100,
            y2=50
        )
        
        assert element.label == "title"
        assert element.x1 == 10
        assert element.y1 == 20
        assert element.x2 == 100
        assert element.y2 == 50
        assert element.text == ""
        assert element.crop_image == ""
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        element = LayoutElement(
            label="heading",
            x1=5,
            y1=10,
            x2=50,
            y2=30,
            text="Test Heading",
            crop_image="base64data"
        )
        
        result = element.to_dict()
        
        assert isinstance(result, dict)
        assert result['label'] == 'heading'
        assert result['x1'] == 5
        assert result['y1'] == 10
        assert result['x2'] == 50
        assert result['y2'] == 30
        assert result['text'] == "Test Heading"
        assert result['crop_image'] == "base64data"
    
    def test_default_optional_fields(self):
        """Test optional fields have defaults."""
        element = LayoutElement(
            label="text",
            x1=0,
            y1=0,
            x2=10,
            y2=10
        )
        
        assert element.text == ""
        assert element.crop_image == ""


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""
    
    def test_initialization(self):
        """Test creating BoundingBox."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 80
    
    def test_width_calculation(self):
        """Test width property calculation."""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=80)
        
        assert bbox.width == 100
    
    def test_height_calculation(self):
        """Test height property calculation."""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=80)
        
        assert bbox.height == 60
    
    def test_area_calculation(self):
        """Test area property calculation."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        
        assert bbox.area == 5000
    
    def test_zero_dimensions(self):
        """Test bbox with zero dimensions."""
        bbox = BoundingBox(x1=10, y1=10, x2=10, y2=10)
        
        assert bbox.width == 0
        assert bbox.height == 0
        assert bbox.area == 0
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        bbox = BoundingBox(x1=5, y1=10, x2=55, y2=60)
        
        result = bbox.to_dict()
        
        assert isinstance(result, dict)
        assert result['x1'] == 5
        assert result['y1'] == 10
        assert result['x2'] == 55
        assert result['y2'] == 60
        assert len(result) == 4
