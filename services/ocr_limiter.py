"""Process-wide request limiter for the shared vLLM OCR server.

Each extraction already limits its own page fan-out. Without this second gate,
four concurrent documents at four pages each create sixteen HTTP requests for
an eight-sequence vLLM engine. vLLM can queue them, but a shared semaphore gives
predictable pressure and lets documents interleave without multiplying clients.
"""

from __future__ import annotations

import asyncio
import weakref
from contextlib import asynccontextmanager

from config.settings import settings

_limiters: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]
] = weakref.WeakKeyDictionary()


def _limiter() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limit = max(1, int(settings.ocr_global_parallelism))
    cached = _limiters.get(loop)
    if cached is None or cached[0] != limit:
        cached = (limit, asyncio.Semaphore(limit))
        _limiters[loop] = cached
    return cached[1]


@asynccontextmanager
async def ocr_request_slot():
    """Hold one process-wide vLLM request slot."""
    async with _limiter():
        yield
