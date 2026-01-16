"""
Unit tests for utils.bbox_utils module.
"""
import pytest
from PIL import Image
from utils.bbox_utils import (
    extract_grounding_references,
    extract_layout_coordinates,
    draw_bounding_boxes,
    normalize_bbox,
    denormalize_bbox
)


class TestExtractGroundingReferences:
    """Tests for extract_grounding_references function."""
    
    def test_extract_single_reference(self):
        """Test parsing single grounding reference."""
        text = "Some text <|ref|>title<|/ref|><|det|>[[100,200,300,400]]<|/det|> more text"
        
        matches = extract_grounding_references(text)
        
        assert len(matches) == 1
        assert 'title' in matches[0][1]
        assert '[[100,200,300,400]]' in matches[0][2]
    
    def test_extract_multiple_references(self):
        """Test parsing multiple grounding references."""
        text = """
        <|ref|>title<|/ref|><|det|>[[10,20,30,40]]<|/det|>
        <|ref|>image<|/ref|><|det|>[[50,60,70,80]]<|/det|>
        """
        
        matches = extract_grounding_references(text)
        
        assert len(matches) == 2
        assert 'title' in matches[0][1]
        assert 'image' in matches[1][1]
    
    def test_extract_empty_text(self):
        """Test with no grounding references."""
        text = "Plain text without grounding tags"
        
        matches = extract_grounding_references(text)
        
        assert len(matches) == 0
    
    def test_extract_with_newlines(self):
        """Test parsing with multiline text."""
        text = """
        <|ref|>heading<|/ref|>
        <|det|>[[100,100,200,150]]<|/det|>
        """
        
        matches = extract_grounding_references(text)
        
        assert len(matches) == 1


class TestExtractLayoutCoordinates:
    """Tests for extract_layout_coordinates function."""
    
    def test_extract_and_scale_coordinates(self):
        """Test coordinate extraction and scaling."""
        text = "<|ref|>title<|/ref|><|det|>[[0,0,999,999]]<|/det|>"
        img_width, img_height = 1000, 800
        
        elements = extract_layout_coordinates(text, img_width, img_height)
        
        assert len(elements) == 1
        assert elements[0]['label'] == 'title'
        assert elements[0]['x1'] == 0
        assert elements[0]['y1'] == 0
        assert abs(elements[0]['x2'] - 1000) < 2  # Allow small rounding
        assert abs(elements[0]['y2'] - 800) < 2
    
    def test_extract_multiple_boxes_same_label(self):
        """Test extracting multiple boxes for same label."""
        text = "<|ref|>text<|/ref|><|det|>[[100,100,200,200],[300,300,400,400]]<|/det|>"
        
        elements = extract_layout_coordinates(text, 1000, 1000)
        
        assert len(elements) == 2
        assert all(elem['label'] == 'text' for elem in elements)
    
    def test_extract_with_invalid_coordinates(self):
        """Test handling invalid coordinates gracefully."""
        text = "<|ref|>title<|/ref|><|det|>invalid_coords<|/det|>"
        
        elements = extract_layout_coordinates(text, 1000, 1000)
        
        # Should not crash, might return empty list
        assert isinstance(elements, list)
    
    def test_normalize_coordinates(self):
        """Test normalization of coordinates."""
        text = "<|ref|>heading<|/ref|><|det|>[[500,500,999,999]]<|/det|>"
        img_width, img_height = 2000, 2000
        
        elements = extract_layout_coordinates(text, img_width, img_height)
        
        assert len(elements) == 1
        # 500/999 * 2000 ≈ 1000
        assert 900 < elements[0]['x1'] < 1100


class TestDrawBoundingBoxes:
    """Tests for draw_bounding_boxes function."""
    
    def test_draw_single_box(self, sample_base64_image):
        """Test drawing a single bounding box."""
        from utils.image_utils import decode_base64_image
        
        img = decode_base64_image(sample_base64_image)
        elements = [{
            'label': 'title',
            'x1': 10,
            'y1': 10,
            'x2': 50,
            'y2': 30
        }]
        
        annotated_img, crops = draw_bounding_boxes(img, elements, extract_images=False)
        
        assert isinstance(annotated_img, Image.Image)
        assert annotated_img.size == img.size
        assert len(crops) == 0
    
    def test_draw_multiple_boxes(self, sample_base64_image):
        """Test drawing multiple bounding boxes."""
        from utils.image_utils import decode_base64_image
        
        img = decode_base64_image(sample_base64_image)
        elements = [
            {'label': 'title', 'x1': 10, 'y1': 10, 'x2': 40, 'y2': 20},
            {'label': 'text', 'x1': 10, 'y1': 25, 'x2': 40, 'y2': 40}
        ]
        
        annotated_img, crops = draw_bounding_boxes(img, elements)
        
        assert isinstance(annotated_img, Image.Image)
    
    def test_extract_image_crops(self, sample_base64_image):
        """Test extracting image regions."""
        from utils.image_utils import decode_base64_image
        
        img = decode_base64_image(sample_base64_image)
        elements = [{
            'label': 'image',
            'x1': 10,
            'y1': 10,
            'x2': 40,
            'y2': 30
        }]
        
        annotated_img, crops = draw_bounding_boxes(img, elements, extract_images=True)
        
        assert len(crops) == 1
        assert isinstance(crops[0], Image.Image)
        assert 'crop_image' in elements[0]


class TestNormalizeBBox:
    """Tests for normalize_bbox function."""
    
    def test_normalize_full_image(self):
        """Test normalizing bbox covering full image."""
        bbox = {'x1': 0, 'y1': 0, 'x2': 1000, 'y2': 800}
        
        normalized = normalize_bbox(bbox, 1000, 800)
        
        assert normalized['x1'] == 0
        assert normalized['y1'] == 0
        assert normalized['x2'] == 999
        assert normalized['y2'] == 999
    
    def test_normalize_half_image(self):
        """Test normalizing bbox covering half image."""
        bbox = {'x1': 0, 'y1': 0, 'x2': 500, 'y2': 400}
        
        normalized = normalize_bbox(bbox, 1000, 800)
        
        assert abs(normalized['x1'] - 0) < 1
        assert abs(normalized['y1'] - 0) < 1
        assert abs(normalized['x2'] - 499) < 2  # 500/1000 * 999
        assert abs(normalized['y2'] - 499) < 2  # 400/800 * 999


class TestDenormalizeBBox:
    """Tests for denormalize_bbox function."""
    
    def test_denormalize_full_range(self):
        """Test denormalizing bbox from full 0-999 range."""
        bbox = {'x1': 0, 'y1': 0, 'x2': 999, 'y2': 999}
        
        denormalized = denormalize_bbox(bbox, 1000, 800)
        
        assert denormalized['x1'] == 0
        assert denormalized['y1'] == 0
        assert abs(denormalized['x2'] - 1000) < 2
        assert abs(denormalized['y2'] - 800) < 2
    
    def test_denormalize_half_range(self):
        """Test denormalizing half range."""
        bbox = {'x1': 0, 'y1': 0, 'x2': 499, 'y2': 499}
        
        denormalized = denormalize_bbox(bbox, 1000, 800)
        
        assert abs(denormalized['x2'] - 500) < 2
        assert abs(denormalized['y2'] - 400) < 2
    
    def test_normalize_denormalize_roundtrip(self):
        """Test normalize then denormalize returns similar values."""
        original = {'x1': 100, 'y1': 200, 'x2': 500, 'y2': 600}
        img_w, img_h = 1000, 800
        
        normalized = normalize_bbox(original, img_w, img_h)
        denormalized = denormalize_bbox(normalized, img_w, img_h)
        
        # Should be close to original (within rounding error)
        for key in ['x1', 'y1', 'x2', 'y2']:
            assert abs(denormalized[key] - original[key]) < 3
