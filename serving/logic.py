"""
OCR processing logic using refactored modular structure.

This module now uses utilities from utils/ and models from core/.
"""

import ast
import base64
import logging
import re
from io import BytesIO
from typing import AsyncGenerator, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from core.constants import DEFAULT_OCR_PARAMS
from core.models import ServicePageResult
from utils.bbox_utils import draw_bounding_boxes, extract_layout_coordinates_v2
from utils.image_utils import decode_base64_image, image_to_base64, render_pdf_page_to_base64
from utils.text_utils import clean_grounding_format

logger = logging.getLogger(__name__)


# A run of one short unit that reaches the end of the output (the last 40
# chars may be a cut-off partial unit).
_TAIL_LOOP_RE = re.compile(r"(.{1,40}?)\1{5,}.{0,40}\Z", re.S)
_SALVAGE_MIN_CHARS = 100
_LOOP_MIN_CHARS = 200


def _is_degenerate(text: str, truncated: bool = True) -> bool:
    """Detect repetition-loop degenerate OCR output (SKIP_REPEAT equivalent).

    Returns True if the tail of the text shows clear repetition patterns
    that indicate the model is stuck in a generation loop.
    Skips detection when content is mostly LaTeX (long $...$ blocks).

    ``truncated=False`` means the model stopped on its own: a loop never
    ends by itself, so a repetitive tail there is page content (arrow
    tables, empty cells, check boxes) and only an instruction echo counts.
    """
    if not text or len(text) < 100:
        return False
    # Model echoing formatting instructions instead of document content
    if re.search(r"Use <code>|Use <pre>|for code blocks", text[:800], re.IGNORECASE):
        return True
    if not truncated:
        return False
    # A loop that runs to the cut is degenerate whatever it repeats — formula
    # tokens or multi-line units (":\n", "---\n\n") the one-line patterns miss.
    loop = _TAIL_LOOP_RE.search(text[-2000:])
    if loop and loop.end() - loop.start() >= _LOOP_MIN_CHARS:
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


def trim_repetition_tail(text: str) -> str | None:
    """Drop the loop a max_tokens-truncated output ends in; None if little is left.

    A truncated output without a trailing loop (a dense table cut short) is
    returned unchanged — a partial page beats a blank one.
    """
    loop = _TAIL_LOOP_RE.search(text)
    kept = text[: loop.start()].rstrip() if loop else text
    # No second _is_degenerate pass: what precedes the loop may legitimately
    # repeat (empty table cells, password boxes) — the loop itself is gone.
    if len(kept) < _SALVAGE_MIN_CHARS:
        return None
    return kept


# ── Tiled OCR (fallback for pages that loop / overflow max_tokens) ─────────
# The page is cut into horizontal bands at white gaps; each band is OCR'd with
# the grounding prompt, its boxes mapped back to page coordinates, and the raw
# outputs concatenated — layout parsing downstream sees one ordinary page.
TILE_RENDER_MAX_SIZE = 2688  # 2x the page render, so a band keeps detail
TILE_MAX_DEPTH = 2  # page → 2 bands → up to 4 bands
_TILE_MIN_BAND_PX = 200
_CUT_SEARCH_RATIO = 0.2  # look for a gap within ±20 % of the band's middle
_BLOCK_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(\[\[.*?\]\])<\|/det\|>", re.S)
_COLUMN_EDGE = 520  # 0–999: a block ending left of this / starting right of 480 is in a column
_TABLE_JOIN_GAP = 60  # 0–999: max vertical gap between two halves of a cut table
_DET_RE = re.compile(r"(<\|det\|>)(\[\[.*?\]\])(<\|/det\|>)", re.S)


def find_cut_row(
    image: Image.Image, y0: int, y1: int, x0: int = 0, x1: Optional[int] = None
) -> int:
    """Row to split ``image[y0:y1]`` at: the middle of the widest white gap near
    the band's middle, else the least-inked row there (never a text line when
    a gap exists)."""
    x1 = image.width if x1 is None else x1
    gray = np.asarray(image.convert("L").crop((x0, y0, x1, y1)))
    ink = (gray < 128).sum(axis=1)
    height = y1 - y0
    mid = height // 2
    lo = max(1, int(mid - height * _CUT_SEARCH_RATIO))
    hi = min(height - 1, int(mid + height * _CUT_SEARCH_RATIO))
    blank = ink[lo:hi] <= max(1, (x1 - x0) // 500)
    best: Optional[Tuple[int, int]] = None  # (run length, -distance to middle) → centre
    best_centre = None
    run_start = None
    for i, is_blank in enumerate(list(blank) + [False]):
        if is_blank and run_start is None:
            run_start = i
        elif not is_blank and run_start is not None:
            centre = lo + (run_start + i - 1) // 2
            key = (i - run_start, -abs(centre - mid))
            if best is None or key > best:
                best, best_centre = key, centre
            run_start = None
    if best_centre is None:
        window = ink[lo:hi]
        best_centre = lo + min(range(len(window)), key=lambda i: (window[i], abs(lo + i - mid)))
    return y0 + int(best_centre)


def remap_band_grounding(text: str, y0: int, y1: int, page_height: int) -> str:
    """Rewrite a full-width band's ``<|det|>`` boxes (0–999 of the band) to 0–999
    of the page."""

    def page_y(value: float) -> int:
        return round((y0 + value / 999.0 * (y1 - y0)) / page_height * 999)

    def rewrite(match: re.Match) -> str:
        try:
            boxes = ast.literal_eval(match.group(2))
        except (ValueError, SyntaxError):
            return match.group(0)
        mapped = ", ".join(
            f"[{int(bx1)}, {page_y(by1)}, {int(bx2)}, {page_y(by2)}]"
            for bx1, by1, bx2, by2 in boxes
        )
        return f"{match.group(1)}[{mapped}]{match.group(3)}"

    return _DET_RE.sub(rewrite, text)


def _band_base64(image: Image.Image, box: Tuple[int, int, int, int]) -> str:
    band = image.crop(box).convert("RGB")
    max_size = DEFAULT_OCR_PARAMS["max_image_size"]
    if max(band.size) > max_size:
        band.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = BytesIO()
    band.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


async def _ocr_image_once(client, img_b64: str, **kwargs) -> Tuple[str, Optional[str]]:
    """One non-streaming OCR request → (raw grounded output, finish_reason)."""
    from config.settings import settings

    response = await client.chat.completions.create(
        model=kwargs.get("model", settings.vllm_model),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": kwargs.get("prompt", settings.ocr_prompt)},
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
    return response.choices[0].message.content or "", response.choices[0].finish_reason


async def _ocr_band(
    client, image: Image.Image, box: Tuple[int, int, int, int], depth: int, **kwargs
) -> str:
    """OCR the ``box`` region; split it horizontally while it still loops, then
    keep what precedes the loop. Returns page-coordinate grounded text."""
    x0, y0, x1, y1 = box
    text, finish = await _ocr_image_once(client, _band_base64(image, box), **kwargs)

    def to_page(raw: str) -> str:
        return remap_band_grounding(raw, y0, y1, image.height)

    if not _is_degenerate(text, truncated=finish != "stop"):
        return to_page(text)
    if depth < TILE_MAX_DEPTH and y1 - y0 >= 2 * _TILE_MIN_BAND_PX:
        cut = find_cut_row(image, y0, y1, x0, x1)
        upper = await _ocr_band(client, image, (x0, y0, x1, cut), depth + 1, **kwargs)
        lower = await _ocr_band(client, image, (x0, cut, x1, y1), depth + 1, **kwargs)
        return "\n\n".join(part for part in (upper, lower) if part)
    kept = trim_repetition_tail(text)
    logger.warning(
        "OCR region %s still loops — kept %d chars before the loop", box, len(kept or "")
    )
    return to_page(kept) if kept else ""


def _split_blocks(raw: str) -> Tuple[str, list]:
    """(text before the first block, [{label, box, body}]) of a grounded output."""
    matches = list(_BLOCK_RE.finditer(raw))
    if not matches:
        return raw, []
    blocks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        try:
            box = [int(v) for v in ast.literal_eval(match.group(2))[0]]
        except (ValueError, SyntaxError, IndexError, TypeError):
            box = [0, 0, 999, 999]
        blocks.append({"label": match.group(1), "box": box, "body": raw[match.end() : end]})
    return raw[: matches[0].start()], blocks


def _render_block(block: dict) -> str:
    x1, y1, x2, y2 = block["box"]
    body = block["body"].strip("\n")
    return f"<|ref|>{block['label']}<|/ref|><|det|>[[{x1}, {y1}, {x2}, {y2}]]<|/det|>\n{body}\n"


def _join_cut_tables(blocks: list) -> list:
    """Glue the two halves of a table the band cut split (same span, adjacent)."""
    out: list = []
    for block in blocks:
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev["label"] == block["label"] == "table"
            and 0 <= block["box"][1] - prev["box"][3] <= _TABLE_JOIN_GAP
            and min(prev["box"][2], block["box"][2]) - max(prev["box"][0], block["box"][0])
            >= 0.8 * min(prev["box"][2] - prev["box"][0], block["box"][2] - block["box"][0])
        ):
            head = re.sub(r"</table>\s*$", "", prev["body"].rstrip())
            tail = re.sub(r"^\s*<table>", "", block["body"].lstrip())
            px1, py1, px2, _ = prev["box"]
            bx1, _, bx2, by2 = block["box"]
            out[-1] = {
                "label": "table",
                "box": [min(px1, bx1), py1, max(px2, bx2), by2],
                "body": head + tail,
            }
            continue
        out.append(block)
    return out


def _reading_order(blocks: list) -> list:
    """Column-major order for a two-column page (the bands are read top to
    bottom, which interleaves the columns); unchanged otherwise."""
    left = [b for b in blocks if b["box"][2] <= _COLUMN_EDGE]
    right = [b for b in blocks if b["box"][0] >= 999 - _COLUMN_EDGE and b not in left]
    if len(left) < 2 or len(right) < 2:
        return blocks
    ordered: list = []
    segment: list = []
    for block in sorted(blocks, key=lambda b: b["box"][1]):
        if block in left or block in right:
            segment.append(block)
            continue
        ordered += [b for b in segment if b in left] + [b for b in segment if b in right]
        segment = []
        ordered.append(block)  # full-width block: a heading / figure across both columns
    return ordered + [b for b in segment if b in left] + [b for b in segment if b in right]


async def ocr_page_tiled(client, image: Image.Image, **kwargs) -> Optional[str]:
    """Raw grounded output for a whole page OCR'd in bands; None if empty.

    Bands are cut at white rows (never through a table column). The merged
    blocks are put back in reading order and cut tables are rejoined."""
    width, height = image.size
    cut = find_cut_row(image, 0, height)
    parts = [
        await _ocr_band(client, image, box, 1, **kwargs)
        for box in ((0, 0, width, cut), (0, cut, width, height))
    ]
    merged = "\n\n".join(part for part in parts if part)
    if not merged:
        return None
    prefix, blocks = _split_blocks(merged)
    if not blocks:
        return merged
    blocks = _reading_order(_join_cut_tables(blocks))
    return prefix + "".join(_render_block(b) for b in blocks)


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
    finish_reason = None

    if kwargs.get("tiled"):
        hires = decode_base64_image(
            render_pdf_page_to_base64(
                pdf_path,
                page_num,
                target_dpi=2 * kwargs.get("target_dpi", DEFAULT_OCR_PARAMS["target_dpi"]),
                max_size=TILE_RENDER_MAX_SIZE,
            )
            if is_pdf
            else image_to_base64(pdf_path, max_size=TILE_RENDER_MAX_SIZE)
        )
        band_kwargs = {k: v for k, v in kwargs.items() if k not in ("tiled", "max_tokens")}
        tiled = await ocr_page_tiled(client, hires, **band_kwargs)
        if tiled is None:
            yield {
                "type": "error",
                "code": "degenerate",
                "message": "Degenerate OCR output detected (repetition loop in every band)",
            }
            return
        model_response, finish_reason = tiled, "stop"
    elif stream_enabled:
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
            finish_reason = chunk.choices[0].finish_reason or finish_reason
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                model_response += token
                yield {"type": "content", "text": token}
    else:
        model_response, finish_reason = await _ocr_image_once(client, img_b64, **kwargs)

    if _is_degenerate(model_response, truncated=finish_reason != "stop"):
        salvaged = (
            trim_repetition_tail(model_response) if kwargs.get("salvage_repetition") else None
        )
        if salvaged is None:
            yield {
                "type": "error",
                "code": "degenerate",
                "message": "Degenerate OCR output detected (repetition loop)",
            }
            return
        logger.warning(
            "OCR page %s looped to max_tokens — kept %d of %d chars before the loop",
            page_num,
            len(salvaged),
            len(model_response),
        )
        model_response = salvaged

    # Extract layout coordinates using V2 (with full text extraction)
    layout_elements = extract_layout_coordinates_v2(
        model_response,  # Raw response with grounding tags
        img_width,
        img_height,
        page_number=page_num,
    )

    # Draw bounding boxes and extract image crops using utils
    try:
        annotated_img, crops = draw_bounding_boxes(image, layout_elements, extract_images=True)
    except Exception:
        # Visualization must not discard a page whose OCR text already exists.
        annotated_img, crops = image, []

    # Convert annotated image to base64
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
