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


class TestStallDetection:
    """A timer-driven heartbeat proves the *process* is alive, not that the
    *work* is progressing — a stage stuck in a loop of returning-but-useless
    LLM calls looked perfectly healthy to Temporal for the full 12h timeout.

    Opt-in only: BUILD_TREE reports no progress at all, so applying this
    everywhere would kill healthy runs.
    """

    @pytest.mark.asyncio
    async def test_keeps_heartbeating_while_probe_changes(self, monkeypatch):
        calls = []
        monkeypatch.setattr(da.activity, "heartbeat", lambda *a, **kw: calls.append(1))
        monkeypatch.setattr(da, "_HEARTBEAT_INTERVAL_SECONDS", 0.01)

        ticks = iter(range(1000))

        async def _work():
            await asyncio.sleep(0.12)
            return "ok"

        result = await da._with_heartbeat(
            _work(), stall_probe=lambda: next(ticks), stall_timeout=0.05
        )
        assert result == "ok"
        assert len(calls) >= 3

    @pytest.mark.asyncio
    async def test_stops_heartbeating_when_probe_is_frozen(self, monkeypatch):
        calls = []
        monkeypatch.setattr(da.activity, "heartbeat", lambda *a, **kw: calls.append(1))
        monkeypatch.setattr(da, "_HEARTBEAT_INTERVAL_SECONDS", 0.01)

        async def _work():
            await asyncio.sleep(0.3)
            return "ok"

        task = asyncio.ensure_future(
            da._with_heartbeat(_work(), stall_probe=lambda: "frozen", stall_timeout=0.05)
        )
        await asyncio.sleep(0.15)
        frozen_count = len(calls)
        await asyncio.sleep(0.1)
        # Heartbeats must have stopped so Temporal's heartbeat_timeout fires.
        assert len(calls) == frozen_count
        await task

    @pytest.mark.asyncio
    async def test_probe_failure_keeps_heartbeating(self, monkeypatch):
        """Fail open: a broken probe must never be the reason a healthy
        long-running stage gets killed."""
        calls = []
        monkeypatch.setattr(da.activity, "heartbeat", lambda *a, **kw: calls.append(1))
        monkeypatch.setattr(da, "_HEARTBEAT_INTERVAL_SECONDS", 0.01)

        def _broken():
            raise RuntimeError("db down")

        await da._with_heartbeat(
            asyncio.sleep(0.1, result=None), stall_probe=_broken, stall_timeout=0.02
        )
        assert len(calls) >= 3

    @pytest.mark.asyncio
    async def test_no_probe_preserves_old_behaviour(self, monkeypatch):
        calls = []
        monkeypatch.setattr(da.activity, "heartbeat", lambda *a, **kw: calls.append(1))
        monkeypatch.setattr(da, "_HEARTBEAT_INTERVAL_SECONDS", 0.01)

        await da._with_heartbeat(asyncio.sleep(0.1))
        assert len(calls) >= 3
