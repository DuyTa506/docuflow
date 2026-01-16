"""
Unit tests for core.constants module.
"""
import re
import pytest
from core.constants import (
    LABEL_HIERARCHY_WEIGHTS,
    DEFAULT_SPATIAL_WEIGHTS,
    HIERARCHY_THRESHOLDS,
    OCR_PROMPTS,
    DEFAULT_OCR_PARAMS,
    GROUNDING_PATTERN
)


class TestLabelHierarchyWeights:
    """Tests for LABEL_HIERARCHY_WEIGHTS constant."""
    
    def test_weights_exist(self):
        """Test all expected labels have weights."""
        expected_labels = [
            'title', 'sub_title', 'heading', 'text',
            'table', 'image', 'footer', 'caption'
        ]
        
        for label in expected_labels:
            assert label in LABEL_HIERARCHY_WEIGHTS
    
    def test_weights_valid_range(self):
        """Test weights are between 0 and 1."""
        for label, weight in LABEL_HIERARCHY_WEIGHTS.items():
            assert 0.0 <= weight <= 1.0, f"{label} has invalid weight: {weight}"
    
    def test_title_highest_weight(self):
        """Test title has the highest weight."""
        assert LABEL_HIERARCHY_WEIGHTS['title'] == 1.0
    
    def test_footer_low_weight(self):
        """Test footer has low weight."""
        assert LABEL_HIERARCHY_WEIGHTS['footer'] < 0.2


class TestDefaultSpatialWeights:
    """Tests for DEFAULT_SPATIAL_WEIGHTS constant."""
    
    def test_weights_sum_to_one(self):
        """Test spatial weights sum to 1.0."""
        total = sum(DEFAULT_SPATIAL_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01  # Allow small floating point error
    
    def test_required_keys_exist(self):
        """Test all required keys exist."""
        required_keys = ['vertical', 'size', 'label', 'indent']
        
        for key in required_keys:
            assert key in DEFAULT_SPATIAL_WEIGHTS
    
    def test_label_weight_highest(self):
        """Test label weight is the strongest signal."""
        assert DEFAULT_SPATIAL_WEIGHTS['label'] >= max(
            w for k, w in DEFAULT_SPATIAL_WEIGHTS.items() if k != 'label'
        )


class TestHierarchyThresholds:
    """Tests for HIERARCHY_THRESHOLDS constant."""
    
    def test_thresholds_ordered(self):
        """Test thresholds are in descending order."""
        levels = sorted(HIERARCHY_THRESHOLDS.keys())
        
        for i in range(len(levels) - 1):
            assert HIERARCHY_THRESHOLDS[levels[i]] >= HIERARCHY_THRESHOLDS[levels[i+1]]
    
    def test_all_levels_present(self):
        """Test all expected levels exist."""
        expected_levels = [0, 1, 2, 3, 4, 5]
        
        for level in expected_levels:
            assert level in HIERARCHY_THRESHOLDS


class TestOCRPrompts:
    """Tests for OCR_PROMPTS constant."""
    
    def test_prompts_completeness(self):
        """Test all prompt types exist."""
        expected_prompts = ['markdown', 'free_ocr', 'describe']
        
        for prompt_type in expected_prompts:
            assert prompt_type in OCR_PROMPTS
    
    def test_prompts_not_empty(self):
        """Test prompts are not empty strings."""
        for prompt_type, prompt in OCR_PROMPTS.items():
            assert len(prompt) > 0, f"{prompt_type} prompt is empty"
    
    def test_prompts_contain_image_tag(self):
        """Test prompts contain <image> tag."""
        for prompt_type, prompt in OCR_PROMPTS.items():
            assert '<image>' in prompt, f"{prompt_type} missing <image> tag"


class TestDefaultOCRParams:
    """Tests for DEFAULT_OCR_PARAMS constant."""
    
    def test_required_params_exist(self):
        """Test all required parameters exist."""
        required_params = [
            'max_tokens', 'temperature', 
            'target_dpi', 'max_image_size'
        ]
        
        for param in required_params:
            assert param in DEFAULT_OCR_PARAMS
    
    def test_params_valid_values(self):
        """Test parameters have valid values."""
        assert DEFAULT_OCR_PARAMS['max_tokens'] > 0
        assert 0.0 <= DEFAULT_OCR_PARAMS['temperature'] <= 2.0
        assert DEFAULT_OCR_PARAMS['target_dpi'] > 0
        assert DEFAULT_OCR_PARAMS['max_image_size'] > 0
    
    def test_temperature_zero(self):
        """Test default temperature is 0 for deterministic output."""
        assert DEFAULT_OCR_PARAMS['temperature'] == 0.0


class TestGroundingPattern:
    """Tests for GROUNDING_PATTERN constant."""
    
    def test_pattern_compiles(self):
        """Test regex pattern compiles successfully."""
        try:
            re.compile(GROUNDING_PATTERN)
        except re.error as e:
            pytest.fail(f"GROUNDING_PATTERN doesn't compile: {e}")
    
    def test_pattern_matches_grounding_format(self):
        """Test pattern matches expected grounding format."""
        test_text = "<|ref|>title<|/ref|><|det|>[[100,200,300,400]]<|/det|>"
        
        matches = re.findall(GROUNDING_PATTERN, test_text)
        
        assert len(matches) > 0
        assert 'title' in matches[0][1]
    
    def test_pattern_extracts_coordinates(self):
        """Test pattern extracts coordinate information."""
        test_text = "<|ref|>image<|/ref|><|det|>[[10,20,30,40]]<|/det|>"
        
        matches = re.findall(GROUNDING_PATTERN, test_text)
        
        assert len(matches) == 1
        assert '[[10,20,30,40]]' in matches[0][2]
