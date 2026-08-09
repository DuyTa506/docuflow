"""Shared helpers for Temporal activities."""

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from temporalio import activity

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_SECONDS = 20


async def _heartbeat_forever(
    stall_probe: Optional[Callable[[], Any]] = None,
    stall_timeout: Optional[float] = None,
) -> None:
    """Ping Temporal until cancelled — or until the work stops progressing.

    Without a probe this reports *process* liveness only: a stage stuck in a
    loop of LLM calls that return but achieve nothing keeps pinging happily,
    and Temporal cannot tell it apart from healthy work until the (12h)
    start_to_close timeout. A probe turns the ping into evidence of progress.
    """
    last_value: Any = object()
    last_change = time.monotonic()

    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)

        if stall_probe is not None and stall_timeout is not None:
            try:
                value = stall_probe()
            except Exception as exc:
                # Fail open: a broken probe must never be the reason a healthy
                # long-running stage is killed.
                logger.debug("Stall probe failed, assuming progress: %s", exc)
                last_change = time.monotonic()
            else:
                if value != last_value:
                    last_value = value
                    last_change = time.monotonic()
                elif time.monotonic() - last_change > stall_timeout:
                    logger.warning(
                        "No progress for %.0fs — stopping heartbeat so Temporal "
                        "fails this attempt instead of waiting out the full timeout",
                        stall_timeout,
                    )
                    return

        activity.heartbeat()


async def _with_heartbeat(
    coro,
    *,
    stall_probe: Optional[Callable[[], Any]] = None,
    stall_timeout: Optional[float] = None,
):
    """Run `coro` while periodically heartbeating, so activities with a
    `heartbeat_timeout` aren't killed just because the wrapped work doesn't
    report its own progress (e.g. long LLM chains on large documents).

    Pass `stall_probe` (any value that changes as work advances, e.g. a task
    row's updated_at) plus `stall_timeout` to also detect a *stalled* stage.
    Only opt in where progress is genuinely reported — a stage that never
    reports would look permanently stalled.
    """
    heartbeat_task = asyncio.create_task(_heartbeat_forever(stall_probe, stall_timeout))
    try:
        return await coro
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
