"""
OCR processing logic using refactored modular structure.

This module now uses utilities from utils/ and models from core/.
"""

import re
from typing import AsyncGenerator, Dict

from core.constants import DEFAULT_OCR_PARAMS
from core.models import ServicePageResult
from utils.bbox_utils import draw_bounding_boxes, extract_layout_coordinates_v2
from utils.image_utils import decode_base64_image, image_to_base64, render_pdf_page_to_base64
from utils.text_utils import clean_grounding_format


def _is_degenerate(text: str) -> bool:
    """Detect repetition-loop degenerate OCR output (SKIP_REPEAT equivalent).

    Returns True if the tail of the text shows clear repetition patterns
    that indicate the model is stuck in a generation loop.
    Skips detection when content is mostly LaTeX (long $...$ blocks).
    """
    if not text or len(text) < 100:
        return False
    # Model echoing formatting instructions instead of document content
    if re.search(r"Use <code>|Use <pre>|for code blocks", text[:800], re.IGNORECASE):
        return True
    # LaTeX-heavy pages: don't treat repeated backslashes as degenerate
    if text.count("$") >= 4 or "\\frac" in text or "\\sum" in text:
        return False
    tail = text[-300:]
    patterns = [
        r"(\d+\.\s*){6,}",  # "1. 2. 3. 4. 5. 6."
        r"(\n\n){8,}",  # excessive blank lines
        r"(.{4,40})\1{5,}",  # any short phrase repeated 5+ times
    ]
    return any(re.search(p, tail) for p in patterns)


async def process_page_api(
    client, pdf_path: str, page_num: int, stream_enabled: bool = True, **kwargs
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
        Dict events with types: 'image', 'content', 'result', 'error'
    """
    # Determine if input is PDF or image
    is_pdf = pdf_path.lower().endswith(".pdf")

    # Render page to base64 using utils
    if is_pdf:
        img_b64 = render_pdf_page_to_base64(
            pdf_path,
            page_num,
            target_dpi=kwargs.get("target_dpi", DEFAULT_OCR_PARAMS["target_dpi"]),
            max_size=kwargs.get("max_size", DEFAULT_OCR_PARAMS["max_image_size"]),
        )
    else:
        img_b64 = image_to_base64(
            pdf_path, max_size=kwargs.get("max_size", DEFAULT_OCR_PARAMS["max_image_size"])
        )

    # Decode to get image dimensions
    image = decode_base64_image(img_b64)
    img_width, img_height = image.size

    # Send image to frontend
    yield {"type": "image", "image_base64": img_b64}

    # Build prompt — driven by settings, not hardcoded
    from config.settings import settings

    _model = kwargs.get("model", settings.vllm_model)
    _prompt = kwargs.get("prompt", settings.ocr_prompt)

    # Call vLLM API
    model_response = ""

    if stream_enabled:
        stream = await client.chat.completions.create(
            model=_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=kwargs.get("max_tokens", DEFAULT_OCR_PARAMS["max_tokens"]),
            temperature=kwargs.get("temperature", DEFAULT_OCR_PARAMS["temperature"]),
            extra_body={
                "skip_special_tokens": False,  # Keep grounding format
                "logits_processors": [
                    {
                        "qualname": "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
                        "kwargs": {
                            "ngram_size": 20,
                            "window_size": 50,
                            "whitelist_token_ids": [128821, 128822],
                        },
                    }
                ],
            },
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                model_response += token
                yield {"type": "content", "text": token}
    else:
        response = await client.chat.completions.create(
            model=_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=kwargs.get("max_tokens", DEFAULT_OCR_PARAMS["max_tokens"]),
            temperature=kwargs.get("temperature", DEFAULT_OCR_PARAMS["temperature"]),
            extra_body={
                "skip_special_tokens": False,
                "logits_processors": [
                    {
                        "qualname": "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
                        "kwargs": {
                            "ngram_size": 20,
                            "window_size": 50,
                            "whitelist_token_ids": [128821, 128822],
                        },
                    }
                ],
            },
            stream=False,
        )
        model_response = response.choices[0].message.content

    if _is_degenerate(model_response):
        yield {"type": "error", "message": "Degenerate OCR output detected (repetition loop)"}
        return

    # Extract layout coordinates using V2 (with full text extraction)
    layout_elements = extract_layout_coordinates_v2(
        model_response,  # Raw response with grounding tags
        img_width,
        img_height,
        page_number=page_num,
    )

    # Draw bounding boxes and extract image crops using utils
    annotated_img, crops = draw_bounding_boxes(image, layout_elements, extract_images=True)

    # Convert annotated image to base64
    import base64
    from io import BytesIO

    buf = BytesIO()
    annotated_img.save(buf, format="PNG")
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
        crops_base64=[
            elem.get("crop_image", "") for elem in layout_elements if elem.get("crop_image")
        ],
    )

    yield {"type": "result", "result": result}


# For backwards compatibility, import old names
from core.models import LayoutElement as LayoutElement
from core.models import ServicePageResult as ServicePageResult
