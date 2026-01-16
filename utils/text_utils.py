"""
Text utilities for OCR workflow.

Handles text cleaning and processing.
"""
import re
from typing import Optional

from core.constants import GROUNDING_PATTERN


def clean_grounding_format(text: str, keep_images: bool = False) -> str:
    """
    Remove grounding format tags from text.
    
    Args:
        text: Text with grounding tags
        keep_images: If True, keep image placeholders
    
    Returns:
        Cleaned markdown text
    """
    if not text:
        return ""
    
    matches = re.findall(GROUNDING_PATTERN, text, re.DOTALL)
    
    img_num = 0
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            if keep_images:
                text = text.replace(match[0], f'\n\n**[Figure {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(match[0], '', 1)
        else:
            # Remove the entire line containing the grounding tag
            text = re.sub(rf'(?m)^[^\n]*{re.escape(match[0])}[^\n]*\n?', '', text)
    
    return text.strip()


def extract_text_from_grounding(grounding_text: str) -> str:
    """
    Extract plain text from grounding format.
    
    Args:
        grounding_text: Text with grounding tags
    
    Returns:
        Plain text without tags
    """
    # Remove all grounding tags
    cleaned = re.sub(GROUNDING_PATTERN, '', grounding_text, flags=re.DOTALL)
    return cleaned.strip()


def strip_markdown_headers(text: str) -> str:
    """
    Remove markdown header symbols from text.
    
    Args:
        text: Markdown text
    
    Returns:
        Text without header symbols
    """
    return re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
