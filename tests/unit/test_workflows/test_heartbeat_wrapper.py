"""_with_heartbeat must keep heartbeating while a long-running activity body
runs, so Temporal's heartbeat_timeout doesn't kill activities on large
documents just because the wrapped work never reports its own progress.
"""

import asyncio

import pytest

from workflows.activities import _common as da


@pytest.mark.asyncio
async def test_with_heartbeat_calls_heartbeat_periodically(monkeypatch):
    calls = []
    monkeypatch.setattr(da.activity, "heartbeat", lambda *a, **kw: calls.append(1))
    monkeypatch.setattr(da, "_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    result = await da._with_heartbeat(asyncio.sleep(0.05, result="done"))

    assert result == "done"
    assert len(calls) >= 1


@pytest.mark.asyncio
async def test_with_heartbeat_stops_after_coro_completes(monkeypatch):
    calls = []
    monkeypatch.setattr(da.activity, "heartbeat", lambda *a, **kw: calls.append(1))
    monkeypatch.setattr(da, "_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    await da._with_heartbeat(asyncio.sleep(0.02))
    count_after_first = len(calls)

    await asyncio.sleep(0.05)
    assert len(calls) == count_after_first, "heartbeat task must be cancelled, not leak"


@pytest.mark.asyncio
async def test_with_heartbeat_propagates_exception(monkeypatch):
    monkeypatch.setattr(da.activity, "heartbeat", lambda *a, **kw: None)
    monkeypatch.setattr(da, "_HEARTBEAT_INTERVAL_SECONDS", 0.01)

    async def _boom():
        raise ValueError("stage failed")

    with pytest.raises(ValueError, match="stage failed"):
        await da._with_heartbeat(_boom())
