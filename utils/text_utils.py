"""
Text utilities for OCR workflow.

Handles text cleaning and processing.
"""

import re

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
        if "<|ref|>image<|/ref|>" in match[0]:
            if keep_images:
                text = text.replace(match[0], f"\n\n**[Figure {img_num + 1}]**\n\n", 1)
                img_num += 1
            else:
                text = text.replace(match[0], "", 1)
        else:
            # Remove the entire line containing the grounding tag
            text = re.sub(rf"(?m)^[^\n]*{re.escape(match[0])}[^\n]*\n?", "", text)

    return text.strip()
