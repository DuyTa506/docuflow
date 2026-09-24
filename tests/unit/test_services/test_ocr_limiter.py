"""All documents share one vLLM request budget."""

import asyncio

import pytest


@pytest.mark.asyncio
async def test_global_ocr_limiter_bounds_cross_document_requests(monkeypatch):
    from services import ocr_limiter

    monkeypatch.setattr(ocr_limiter.settings, "ocr_global_parallelism", 2)
    active = 0
    peak = 0
    lock = asyncio.Lock()

    async def request():
        nonlocal active, peak
        async with ocr_limiter.ocr_request_slot():
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.02)
            async with lock:
                active -= 1

    await asyncio.gather(*(request() for _ in range(8)))
    assert peak == 2
