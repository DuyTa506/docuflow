"""
OCR processing logic using refactored modular structure.

This module now uses utilities from utils/ and models from core/.
"""
from typing import AsyncGenerator, Dict

from core.models import ServicePageResult
from core.constants import DEFAULT_OCR_PARAMS
from utils.image_utils import (
    render_pdf_page_to_base64,
    image_to_base64,
    decode_base64_image
)
from utils.bbox_utils import (
    extract_layout_coordinates,
    draw_bounding_boxes
)
from utils.text_utils import clean_grounding_format


async def process_page_api(
    client,
    pdf_path: str,
    page_num: int,
    stream_enabled: bool = True,
    **kwargs
) -> AsyncGenerator[Dict, None]:
    """
    Process a single page using the DeepSeek OCR vLLM server.
    
    This is an async generator that yields events during processing.
    
    Args:
        client: AsyncOpenAI client instance configured for vLLM server
        pdf_path: Path to PDF or image file
        page_num: 1-indexed page number
        stream_enabled: Whether to stream tokens
        **kwargs: Additional parameters
    
    Yields:
        Dict events with types: 'image', 'content', 'result'
    """
    # Determine if input is PDF or image
    is_pdf = pdf_path.lower().endswith('.pdf')
    
    # Render page to base64 using utils
    if is_pdf:
        img_b64 = render_pdf_page_to_base64(
            pdf_path, 
            page_num,
            target_dpi=kwargs.get('target_dpi', DEFAULT_OCR_PARAMS['target_dpi'])
        )
    else:
        img_b64 = image_to_base64(
            pdf_path,
            max_size=kwargs.get('max_size', DEFAULT_OCR_PARAMS['max_image_size'])
        )
    
    # Decode to get image dimensions
    image = decode_base64_image(img_b64)
    img_width, img_height = image.size
    
    # Send image to frontend
    yield {"type": "image", "image_base64": img_b64}
    
    # Build prompt
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    # Call vLLM API
    model_response = ""
    
    if stream_enabled:
        stream = await client.chat.completions.create(
            model="ocr",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            max_tokens=kwargs.get('max_tokens', DEFAULT_OCR_PARAMS['max_tokens']),
            temperature=kwargs.get('temperature', DEFAULT_OCR_PARAMS['temperature']),
            extra_body={
                "skip_special_tokens": False,  # Keep grounding format
            },
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                model_response += token
                yield {"type": "content", "text": token}
    else:
        response = await client.chat.completions.create(
            model="ocr",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            max_tokens=kwargs.get('max_tokens', DEFAULT_OCR_PARAMS['max_tokens']),
            temperature=kwargs.get('temperature', DEFAULT_OCR_PARAMS['temperature']),
            extra_body={
                "skip_special_tokens": False,
            },
            stream=False
        )
        model_response = response.choices[0].message.content
    
    # Extract layout coordinates using utils
    layout_elements = extract_layout_coordinates(model_response, img_width, img_height)
    
    # Draw bounding boxes and extract image crops using utils
    annotated_img, crops = draw_bounding_boxes(image, layout_elements, extract_images=True)
    
    # Convert annotated image to base64
    import base64
    from io import BytesIO
    buf = BytesIO()
    annotated_img.save(buf, format='PNG')
    annotated_img_b64 = base64.b64encode(buf.getvalue()).decode()
    
    # Clean markdown using utils
    cleaned_markdown = clean_grounding_format(model_response, keep_images=False)
    
    # Create result using core model
    result = ServicePageResult(
        page_num=page_num,
        markdown=cleaned_markdown,
        image_base64=img_b64,
        annotated_image_base64=annotated_img_b64,
        layout_elements=layout_elements,
        crops_base64=[elem.get('crop_image', '') for elem in layout_elements if elem.get('crop_image')]
    )
    
    yield {"type": "result", "result": result}


# For backwards compatibility, import old names
from core.models import ServicePageResult as ServicePageResult
from core.models import LayoutElement as LayoutElement
