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
