"""
Constants and configuration values for OCR workflow.
"""

# Hierarchy weights for grounding labels
LABEL_HIERARCHY_WEIGHTS = {
    'title': 1.0,
    'sub_title': 0.8,
    'subtitle': 0.8,
    'heading': 0.7,
    'header': 0.65,
    'text': 0.3,
    'paragraph': 0.3,
    'table': 0.4,
    'image': 0.4,
    'figure': 0.4,
    'formula': 0.4,
    'equation': 0.4,
    'caption': 0.2,
    'footer': 0.1,
    'page_number': 0.05
}

# Default weights for spatial scoring
DEFAULT_SPATIAL_WEIGHTS = {
    'vertical': 0.2,    # Position on page
    'size': 0.3,        # Element size
    'label': 0.4,       # Label type (strongest signal)
    'indent': 0.1       # Indentation
}

# Hierarchy level thresholds
HIERARCHY_THRESHOLDS = {
    0: 0.8,   # Document title
    1: 0.6,   # Major section
    2: 0.4,   # Subsection
    3: 0.25,  # Subsubsection
    4: 0.15,  # Paragraph
    5: 0.0    # Supporting elements
}

# OCR prompt templates
OCR_PROMPTS = {
    'markdown': '<image>\n<|grounding|>Convert the document to markdown.',
    'free_ocr': '<image>\nFree OCR.',
    'describe': '<image>\nDescribe this image in detail.'
}

# Default OCR parameters
DEFAULT_OCR_PARAMS = {
    'max_tokens': 4096,
    'temperature': 0.0,
    'target_dpi': 200,
    'max_image_size': 2048
}

# Grounding format regex patterns
GROUNDING_PATTERN = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
