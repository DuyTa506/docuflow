"""File-backed GPU leases: acquire, heartbeat, expire, wait."""

import asyncio
import time

import pytest

from services import gpu_lease as gpu_lease_mod
from services.gpu_lease import GpuLeaseBusy, acquire_with_wait, heartbeat, release, try_acquire


@pytest.fixture
def lease_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_lease_mod, "_lease_dir", lambda: tmp_path)
    return tmp_path


def test_acquire_and_release(lease_dir):
    assert try_acquire("docling", "extract:DOC_1", ttl_seconds=30)
    assert not try_acquire("docling", "extract:DOC_2", ttl_seconds=30)
    release("docling", "extract:DOC_1")
    assert try_acquire("docling", "extract:DOC_2", ttl_seconds=30)


def test_same_holder_renews(lease_dir):
    assert try_acquire("docling", "extract:DOC_1", ttl_seconds=30)
    assert heartbeat("docling", "extract:DOC_1", ttl_seconds=30)


def test_expired_lease_can_be_stolen(lease_dir):
    assert try_acquire("docling", "extract:DOC_1", ttl_seconds=1)
    time.sleep(1.1)
    assert try_acquire("docling", "extract:DOC_2", ttl_seconds=30)


def test_wait_times_out(lease_dir):
    assert try_acquire("docling", "extract:DOC_1", ttl_seconds=60)

    async def _wait():
        await acquire_with_wait("docling", "extract:DOC_2", wait_seconds=1, poll_seconds=0.2)

    with pytest.raises(GpuLeaseBusy):
        asyncio.run(_wait())


def test_wait_zero_is_infinite_and_notifies(lease_dir):
    assert try_acquire("docling", "extract:DOC_1", ttl_seconds=60)
    notified = []

    async def _wait():
        async def _run():
            task = asyncio.create_task(
                acquire_with_wait(
                    "docling",
                    "extract:DOC_2",
                    wait_seconds=0,
                    poll_seconds=0.05,
                    on_waiting=lambda: notified.append(True),
                )
            )
            await asyncio.sleep(0.12)
            assert not task.done()
            assert notified == [True]
            release("docling", "extract:DOC_1")
            await asyncio.wait_for(task, timeout=1.0)

        await _run()

    asyncio.run(_wait())
