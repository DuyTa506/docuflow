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

# Default weights for spatial scoring (UPDATED: includes whitespace)
DEFAULT_SPATIAL_WEIGHTS = {
    'label': 0.40,       # Label type (strongest signal)
    'whitespace': 0.25,  # White-space isolation (NEW)
    'size': 0.15,        # Element size
    'vertical': 0.10,    # Position on page
    'indent': 0.10       # Indentation
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

# Zone types for document layout
ZONE_TYPES = [
    'title_block',
    'author_block',
    'abstract',
    'section_heading',
    'main_text',
    'figure',
    'table',
    'caption',
    'equation',
    'footnote',
    'header',
    'footer',
    'page_number',
    'sidebar',
    'unknown'
]

# Zone priority for reading order (lower = read first)
ZONE_PRIORITY = {
    'title_block': 0,
    'author_block': 1,
    'abstract': 2,
    'section_heading': 3,
    'main_text': 4,
    'figure': 5,
    'table': 5,
    'caption': 6,
    'equation': 4,
    'footnote': 8,
    'header': 9,
    'footer': 10,
    'page_number': 10,
    'sidebar': 7,
    'unknown': 5,
}

# Spacing thresholds for hierarchy detection
SPACING_THRESHOLDS = {
    'section_gap_ratio': 2.5,     # Section gap > 2.5x median line height
    'paragraph_gap_ratio': 1.5,   # Paragraph gap > 1.5x median
    'heading_isolation': 2.0,     # Heading white-space > 2x median
    'min_line_height': 10,        # Minimum line height in pixels
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
