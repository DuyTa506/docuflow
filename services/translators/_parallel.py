"""Shared async parallel helper for translators."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

ProgressCallback = Optional[Callable[[int, str], Awaitable[None] | None]]


async def run_parallel(
    items: List[T],
    worker: Callable[[int, T], Awaitable[R]],
    *,
    parallelism: int,
    on_progress: ProgressCallback = None,
    progress_label: str = "Item",
) -> List[R]:
    """Run ``worker`` over ``items`` with bounded concurrency; preserve order."""
    if not items:
        return []

    sem = asyncio.Semaphore(max(1, parallelism))
    results: List[Optional[R]] = [None] * len(items)
    done = 0
    lock = asyncio.Lock()

    async def one(idx: int, item: T) -> None:
        nonlocal done
        async with sem:
            results[idx] = await worker(idx, item)
        async with lock:
            done += 1
            if on_progress:
                pct = int((done / len(items)) * 95)
                msg = f"{progress_label} {done}/{len(items)}"
                await _maybe_await(on_progress(pct, msg))

    await asyncio.gather(*(one(i, item) for i, item in enumerate(items)))
    return results  # type: ignore[return-value]


async def _maybe_await(result):
    if result is not None and hasattr(result, "__await__"):
        await result
