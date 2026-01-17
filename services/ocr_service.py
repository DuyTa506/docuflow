"""
OCR Service - Handles document OCR processing via vLLM.

This service orchestrates the OCR workflow including image processing,
API calls, and result formatting.
"""
import base64
from io import BytesIO
from typing import AsyncGenerator, Dict, Optional

from core.models import ServicePageResult
from core.constants import DEFAULT_OCR_PARAMS, OCR_PROMPTS
from utils.image_utils import render_pdf_page_to_base64, image_to_base64, decode_base64_image
from utils.bbox_utils import extract_layout_coordinates, draw_bounding_boxes
from utils.text_utils import clean_grounding_format


class OCRService:
    """Service for OCR processing using vLLM."""
    
    def __init__(
        self,
        client,
        api_key: Optional[str] = None,
        server_url: Optional[str] = None,
        model: str = "ocr"
    ):
        """
        Initialize OCR service.
        
        Args:
            client: AsyncOpenAI client instance
            api_key: API key for vLLM (optional)
            server_url: vLLM server URL (optional)
            model: Model name (default: "ocr")
        """
        self.client = client
        self.api_key = api_key
        self.server_url = server_url
        self.model = model
    
    async def process_page(
        self,
        file_path: str,
        page_num: int,
        stream: bool = True,
        prompt: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict, None]:
        """
        Process a single page with OCR.
        
        Args:
            file_path: Path to PDF or image file
            page_num: 1-indexed page number
            stream: Whether to stream tokens
            prompt: Custom prompt (default: markdown conversion)
            **kwargs: Additional OCR parameters
        
        Yields:
            Dict events with types: 'image', 'content', 'result'
        """
        # Determine file type
        is_pdf = file_path.lower().endswith('.pdf')
        
        # Render to base64
        if is_pdf:
            img_b64 = render_pdf_page_to_base64(
                file_path,
                page_num,
                target_dpi=kwargs.get('target_dpi', DEFAULT_OCR_PARAMS['target_dpi'])
            )
        else:
            img_b64 = image_to_base64(
                file_path,
                max_size=kwargs.get('max_size', DEFAULT_OCR_PARAMS['max_image_size'])
            )
        
        # Get image dimensions
        image = decode_base64_image(img_b64)
        img_width, img_height = image.size
        
        # Yield image
        yield {"type": "image", "image_base64": img_b64}
        
        # Use default prompt if not provided
        if prompt is None:
            prompt = OCR_PROMPTS['markdown']
        
        # Call vLLM
        model_response = await self._call_vllm(
            img_b64=img_b64,
            prompt=prompt,
            stream=stream,
            **kwargs
        )
        
        # Stream tokens if enabled
        if stream:
            full_response = ""
            async for chunk in model_response:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield {"type": "content", "text": token}
            model_response = full_response
        else:
            model_response = model_response.choices[0].message.content
        
        # Process result
        result = self._process_ocr_result(
            model_response,
            page_num,
            img_b64,
            image,
            img_width,
            img_height
        )
        
        yield {"type": "result", "result": result}
    
    async def _call_vllm(
        self,
        img_b64: str,
        prompt: str,
        stream: bool,
        **kwargs
    ):
        """Call vLLM API with image and prompt."""
        return await self.client.chat.completions.create(
            model=self.model,
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
            stream=stream
        )
    
    def _process_ocr_result(
        self,
        model_response: str,
        page_num: int,
        img_b64: str,
        image,
        img_width: int,
        img_height: int
    ) -> ServicePageResult:
        """Process OCR result and extract metadata."""
        
        # DEBUG: Check if model_response has grounding tags
        print(f"[DEBUG OCR] model_response length: {len(model_response)}")
        print(f"[DEBUG OCR] First 300 chars: {model_response[:300]}")
        print(f"[DEBUG OCR] Contains '<|ref|>': {('<|ref|>' in model_response)}")
        print(f"[DEBUG OCR] Contains '<|det|>': {('<|det|>' in model_response)}")
        
        # Extract layout coordinates WITH FULL TEXT (V2)
        from utils.bbox_utils import extract_layout_coordinates_v2
        
        layout_elements = extract_layout_coordinates_v2(
            model_response,  # Raw output with grounding tags
            img_width,
            img_height,
            page_number=page_num
        )
        
        # Draw bounding boxes
        annotated_img, crops = draw_bounding_boxes(
            image,
            layout_elements,
            extract_images=True
        )
        
        # Convert annotated image to base64
        buf = BytesIO()
        annotated_img.save(buf, format='PNG')
        annotated_img_b64 = base64.b64encode(buf.getvalue()).decode()
        
        # Clean markdown
        cleaned_markdown = clean_grounding_format(model_response, keep_images=False)
        
        # Create result
        return ServicePageResult(
            page_num=page_num,
            markdown=cleaned_markdown,
            image_base64=img_b64,
            annotated_image_base64=annotated_img_b64,
            layout_elements=layout_elements,
            crops_base64=[
                elem.get('crop_image', '')
                for elem in layout_elements
                if elem.get('crop_image')
            ]
        )
