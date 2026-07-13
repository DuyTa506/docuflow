"""Shared helpers for Temporal activities."""

import asyncio

from temporalio import activity

_HEARTBEAT_INTERVAL_SECONDS = 20


async def _heartbeat_forever() -> None:
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        activity.heartbeat()


async def _with_heartbeat(coro):
    """Run `coro` while periodically heartbeating, so activities with a
    `heartbeat_timeout` aren't killed just because the wrapped work doesn't
    report its own progress (e.g. long LLM chains on large documents)."""
    heartbeat_task = asyncio.create_task(_heartbeat_forever())
    try:
        return await coro
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
