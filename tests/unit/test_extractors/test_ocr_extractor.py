"""Degenerate OCR is skippable; vLLM infra errors are not."""

from unittest.mock import MagicMock, patch

import pytest

from services.extractors.ocr_extractor import (
    DEGENERATE_RETRY_TEMPERATURE,
    DegenerateOcrError,
    OcrExtractor,
    is_degenerate_ocr_error_event,
)


def _result_event():
    result = MagicMock()
    result.layout_elements = [
        {
            "label": "text",
            "text_full": "ok",
            "bbox_x1": 0,
            "bbox_y1": 0,
            "bbox_x2": 1,
            "bbox_y2": 1,
        }
    ]
    return {"type": "result", "result": result}


def test_is_degenerate_ocr_error_event_matches_code_and_message():
    assert is_degenerate_ocr_error_event({"code": "degenerate", "message": "x"})
    assert is_degenerate_ocr_error_event(
        {"message": "Degenerate OCR output detected (repetition loop)"}
    )
    assert not is_degenerate_ocr_error_event({"message": "vLLM connection refused"})


@pytest.mark.asyncio
async def test_degenerate_event_raises_degenerate_ocr_error():
    async def gen(*_a, **_k):
        yield {
            "type": "error",
            "code": "degenerate",
            "message": "Degenerate OCR output detected (repetition loop)",
        }

    extractor = OcrExtractor(client=object(), file_path="/tmp/x.pdf")
    with patch("serving.logic.process_page_api", gen):
        with pytest.raises(DegenerateOcrError, match="Degenerate"):
            await extractor.extract_page(3, retry_degenerate=False)


@pytest.mark.asyncio
async def test_infra_error_stays_runtimeerror():
    async def gen(*_a, **_k):
        yield {"type": "error", "message": "vLLM connection refused"}

    extractor = OcrExtractor(client=object(), file_path="/tmp/x.pdf")
    with patch("serving.logic.process_page_api", gen):
        with pytest.raises(RuntimeError) as exc:
            await extractor.extract_page(1)
    assert "vLLM" in str(exc.value)
    assert not isinstance(exc.value, DegenerateOcrError)


@pytest.mark.asyncio
async def test_degenerate_retries_once_with_temperature_then_succeeds():
    calls = []

    async def gen(*_a, **kwargs):
        calls.append(kwargs.get("temperature"))
        if len(calls) == 1:
            yield {
                "type": "error",
                "code": "degenerate",
                "message": "Degenerate OCR output detected (repetition loop)",
            }
            return
        yield _result_event()

    extractor = OcrExtractor(client=object(), file_path="/tmp/x.pdf")
    with patch("serving.logic.process_page_api", gen):
        elements = await extractor.extract_page(1)

    assert calls == [None, DEGENERATE_RETRY_TEMPERATURE]
    assert elements[0].text == "ok"


@pytest.mark.asyncio
async def test_degenerate_retry_still_raises_after_second_loop():
    async def gen(*_a, **_k):
        yield {
            "type": "error",
            "code": "degenerate",
            "message": "Degenerate OCR output detected (repetition loop)",
        }

    extractor = OcrExtractor(client=object(), file_path="/tmp/x.pdf")
    with patch("serving.logic.process_page_api", gen):
        with pytest.raises(DegenerateOcrError):
            await extractor.extract_page(1)
