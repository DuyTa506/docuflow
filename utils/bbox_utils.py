"""
Bounding box utilities for OCR workflow.

Handles bounding box extraction, parsing, and visualization.
"""
import base64
import re
from io import BytesIO
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.constants import GROUNDING_PATTERN


def extract_grounding_references(text: str) -> List[tuple]:
    """
    Extract grounding references in the format <|ref|>label<|/ref|><|det|>[[coords]]<|/det|>.
    
    Args:
        text: Text containing grounding references
    
    Returns:
        List of tuples: (full_match, label, coords_str)
    """
    return re.findall(GROUNDING_PATTERN, text, re.DOTALL)


def extract_layout_coordinates(
    text: str,
    img_width: int,
    img_height: int
) -> List[Dict]:
    """
    Parse layout coordinates from DeepSeek OCR output.
    
    DEPRECATED: Use extract_layout_coordinates_v2 for full text extraction.
    
    Args:
        text: OCR output text with grounding format
        img_width: Image width for scaling normalized coordinates
        img_height: Image height for scaling normalized coordinates
    
    Returns:
        List of dicts with label, bbox, and text_content
    """
    refs = extract_grounding_references(text)
    layout_elements = []
    
    for ref in refs:
        label = ref[1].strip()
        coords_str = ref[2]
        
        try:
            # Parse coordinates [[x1,y1,x2,y2]]
            coords = eval(coords_str)
            
            for box in coords:
                x1, y1, x2, y2 = box
                
                # Scale from normalized 0-999 to actual pixels
                px1 = int(x1 / 999.0 * img_width)
                py1 = int(y1 / 999.0 * img_height)
                px2 = int(x2 / 999.0 * img_width)
                py2 = int(y2 / 999.0 * img_height)
                
                layout_elements.append({
                    'label': label,
                    'x1': px1,
                    'y1': py1,
                    'x2': px2,
                    'y2': py2,
                    'text': '',  # Will be filled from markdown
                    'crop_image': ''
                })
        except Exception as e:
            print(f"Warning: Could not parse coordinates for label '{label}': {e}")
            continue
    
    return layout_elements


def extract_header_text(text_segment: str, label: str) -> str:
    """
    Extract clean header/label text from markdown text segment.
    
    Handles:
    - Markdown syntax: "# Title" → "Title"
    - HTML tags: "<center>Figure 1</center>" → "Figure 1"
    - Plain text
    
    Args:
        text_segment: Text segment after grounding tag
        label: Grounding label for context
    
    Returns:
        Cleaned header text
    """
    text = text_segment.strip()
    
    if not text:
        return label  # Fallback to label
    
    # Get first line (headers usually on first line)
    first_line = text.split('\n')[0].strip()
    
    # Remove markdown syntax
    first_line = re.sub(r'^#{1,6}\s+', '', first_line)
    
    # Remove HTML tags
    first_line = re.sub(r'</?center>', '', first_line, flags=re.IGNORECASE)
    first_line = re.sub(r'</?b>', '', first_line, flags=re.IGNORECASE)
    first_line = re.sub(r'</?i>', '', first_line, flags=re.IGNORECASE)
    first_line = re.sub(r'<[^>]+>', '', first_line)
    
    cleaned = first_line.strip()
    
    return cleaned if cleaned else label


def extract_layout_coordinates_v2(
    text: str,
    img_width: int,
    img_height: int,
    page_number: int = 1
) -> List[Dict]:
    """
    Parse layout coordinates WITH FULL TEXT CONTENT from DeepSeek OCR output.
    
    This is the V2 version that enriches elements with:
    - text_content: Clean header/label text
    - text_full: Full text segment until next grounding tag
    
    Args:
        text: OCR output text with grounding format (raw)
        img_width: Image width for scaling normalized coordinates
        img_height: Image height for scaling normalized coordinates
        page_number: Page number for this text
    
    Returns:
        List of enriched dicts with label, bbox, text_content, text_full
    """
    refs = extract_grounding_references(text)
    layout_elements = []
    
    print(f"[DEBUG V2] Found {len(refs)} grounding references")
    print(f"[DEBUG V2] Raw text length: {len(text)}")
    print(f"[DEBUG V2] First 200 chars of text: {text[:200]}")
    
    for i, ref in enumerate(refs):
        label = ref[1].strip()
        coords_str = ref[2]
        full_match = ref[0]  # Full grounding tag
        
        try:
            # Parse coordinates [[x1,y1,x2,y2]]
            coords = eval(coords_str)
            
            # Find text content between this tag and next tag
            match_pos = text.find(full_match)
            
            if i < len(refs) - 1:
                # Not last element - find next grounding tag
                next_match = refs[i + 1][0]
                next_match_pos = text.find(next_match, match_pos + len(full_match))
            else:
                # Last element - take rest of text
                next_match_pos = len(text)
            
            # Extract text segment
            text_start = match_pos + len(full_match)
            text_segment = text[text_start:next_match_pos].strip()
            
            print(f"[DEBUG V2] Element {i}: label={label}, text_segment='{text_segment[:100] if text_segment else 'EMPTY'}...'")
            
            # Extract clean header text
            text_content = extract_header_text(text_segment, label)
            
            print(f"[DEBUG V2] After extract_header_text: text_content='{text_content[:100] if text_content else 'EMPTY'}...'")
            
            for box in coords:
                x1, y1, x2, y2 = box
                
                # Scale from normalized 0-999 to actual pixels
                px1 = int(x1 / 999.0 * img_width)
                py1 = int(y1 / 999.0 * img_height)
                px2 = int(x2 / 999.0 * img_width)
                py2 = int(y2 / 999.0 * img_height)
                
                layout_elements.append({
                    'label': label,
                    'bbox_x1': px1,
                    'bbox_y1': py1,
                    'bbox_x2': px2,
                    'bbox_y2': py2,
                    'x1': px1,  # Backward compat
                    'y1': py1,
                    'x2': px2,
                    'y2': py2,
                    'text_content': text_content,  # NEW: Clean header
                    'text_full': text_segment,      # NEW: Full segment
                    'text': text_content,           # Backward compat
                    'page_number': page_number,
                    'crop_image': ''
                })
        except Exception as e:
            print(f"Warning: Could not parse coordinates for label '{label}': {e}")
            continue
    
    return layout_elements


def draw_bounding_boxes(
    image: Image.Image,
    layout_elements: List[Dict],
    extract_images: bool = True
) -> Tuple[Image.Image, List[Image.Image]]:
    """
    Draw bounding boxes on image and optionally extract image regions.
    
    Args:
        image: PIL Image to draw on
        layout_elements: List of layout element dicts
        extract_images: Whether to crop and return image regions
    
    Returns:
        Tuple of (annotated image, list of cropped images)
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    crops = []
    color_map = {}
    np.random.seed(42)
    
    for elem in layout_elements:
        label = elem['label']
        
        # Assign consistent color per label
        if label not in color_map:
            color_map[label] = (
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            )
        
        color = color_map[label]
        color_a = color + (60,)  # Semi-transparent for overlay
        
        x1, y1, x2, y2 = elem['x1'], elem['y1'], elem['x2'], elem['y2']
        
        # Extract image crops if requested
        if extract_images and label.lower() == 'image':
            try:
                crop = image.crop((x1, y1, x2, y2))
                crops.append(crop)
                
                # Convert to base64
                buf = BytesIO()
                crop.save(buf, format='PNG')
                elem['crop_image'] = base64.b64encode(buf.getvalue()).decode()
            except Exception as e:
                print(f"Warning: Could not crop image: {e}")
        
        # Draw box
        width = 5 if label == 'title' else 3
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw2.rectangle([x1, y1, x2, y2], fill=color_a)
        
        # Draw label
        try:
            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
        except:
            tw, th = len(label) * 10, 15
        
        ty = max(0, y1 - th - 4)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
        draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))
    
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, crops


def normalize_bbox(bbox: Dict, img_width: int, img_height: int) -> Dict:
    """
    Normalize bounding box coordinates to 0-999 range.
    
    Args:
        bbox: Dict with x1, y1, x2, y2 in pixels
        img_width: Image width
        img_height: Image height
    
    Returns:
        Dict with normalized coordinates
    """
    return {
        'x1': int(bbox['x1'] / img_width * 999),
        'y1': int(bbox['y1'] / img_height * 999),
        'x2': int(bbox['x2'] / img_width * 999),
        'y2': int(bbox['y2'] / img_height * 999)
    }


def denormalize_bbox(bbox: Dict, img_width: int, img_height: int) -> Dict:
    """
    Convert normalized (0-999) bounding box to pixel coordinates.
    
    Args:
        bbox: Dict with normalized x1, y1, x2, y2
        img_width: Image width
        img_height: Image height
    
    Returns:
        Dict with pixel coordinates
    """
    return {
        'x1': int(bbox['x1'] / 999.0 * img_width),
        'y1': int(bbox['y1'] / 999.0 * img_height),
        'x2': int(bbox['x2'] / 999.0 * img_width),
        'y2': int(bbox['y2'] / 999.0 * img_height)
    }
