"""Host-level GPU resource leases with TTL heartbeat (single-machine, file-backed).

Docling, vLLM OCR and llama.cpp share one GPU. The lease does not start or stop
those servers — it serializes the *job* that loads extra VRAM (Docling) so a
second extraction cannot pile models on top of a live OCR/digest run after a
crash left the previous holder dead.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Awaitable, Union

from config.capacity import capacity_profile
from config.settings import settings

logger = logging.getLogger(__name__)

RESOURCE_DOCLING = "docling"


class GpuLeaseBusy(Exception):
    """Raised when a resource cannot be acquired before the wait budget expires."""


@dataclass
class LeaseSnapshot:
    resource: str
    holder: str
    expires_at: float
    heartbeat_at: float


def _lease_dir() -> Path:
    configured = (settings.gpu_lease_dir or "").strip()
    root = Path(configured) if configured else Path(settings.upload_dir) / ".gpu-leases"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _path(resource: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in resource) or "gpu"
    return _lease_dir() / f"{safe}.json"


def _read_unlocked(fd: int) -> Optional[dict]:
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, 65536)
    if not raw.strip():
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_unlocked(fd: int, payload: dict) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, encoded)
    os.fsync(fd)


def _open_locked(resource: str):
    path = _path(resource)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def try_acquire(resource: str, holder: str, *, ttl_seconds: Optional[int] = None) -> bool:
    """Take or renew ``resource`` for ``holder``. False if another live holder has it."""
    cap = capacity_profile()
    ttl = float(ttl_seconds if ttl_seconds is not None else cap.gpu_lease_ttl_seconds)
    now = time.time()
    fd = _open_locked(resource)
    try:
        current = _read_unlocked(fd)
        if current:
            other = str(current.get("holder") or "")
            expires = float(current.get("expires_at") or 0)
            if other and other != holder and expires > now:
                return False
        _write_unlocked(
            fd,
            {
                "resource": resource,
                "holder": holder,
                "expires_at": now + ttl,
                "heartbeat_at": now,
            },
        )
        return True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def heartbeat(resource: str, holder: str, *, ttl_seconds: Optional[int] = None) -> bool:
    """Extend TTL if we still own the lease. False if stolen or expired."""
    return try_acquire(resource, holder, ttl_seconds=ttl_seconds)


def release(resource: str, holder: str) -> None:
    fd = _open_locked(resource)
    try:
        current = _read_unlocked(fd)
        if not current or str(current.get("holder") or "") != holder:
            return
        _write_unlocked(fd, {})
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def snapshot(resource: str) -> Optional[LeaseSnapshot]:
    path = _path(resource)
    if not path.is_file():
        return None
    fd = _open_locked(resource)
    try:
        current = _read_unlocked(fd)
        if not current or not current.get("holder"):
            return None
        return LeaseSnapshot(
            resource=resource,
            holder=str(current.get("holder") or ""),
            expires_at=float(current.get("expires_at") or 0),
            heartbeat_at=float(current.get("heartbeat_at") or 0),
        )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


async def acquire_with_wait(
    resource: str,
    holder: str,
    *,
    wait_seconds: Optional[int] = None,
    ttl_seconds: Optional[int] = None,
    poll_seconds: float = 2.0,
    on_waiting: Optional[Callable[[], Union[None, Awaitable[None]]]] = None,
) -> None:
    """Wait until the lease is free.

    ``wait_seconds`` / profile ``gpu_lease_wait_seconds`` <= 0 means wait
    indefinitely (activity heartbeats keep Temporal alive). A positive budget
    still raises ``GpuLeaseBusy`` when exceeded.
    """
    cap = capacity_profile()
    budget = float(wait_seconds if wait_seconds is not None else cap.gpu_lease_wait_seconds)
    infinite = budget <= 0
    deadline = None if infinite else time.monotonic() + budget
    waiting_notified = False
    while True:
        if try_acquire(resource, holder, ttl_seconds=ttl_seconds):
            logger.info("GPU lease acquired resource=%s holder=%s", resource, holder)
            return
        if not waiting_notified and on_waiting is not None:
            waiting_notified = True
            result = on_waiting()
            if asyncio.iscoroutine(result):
                await result
        if deadline is not None and time.monotonic() >= deadline:
            raise GpuLeaseBusy(
                f"GPU resource {resource!r} is busy (holder wait exceeded {budget:.0f}s)"
            )
        await asyncio.sleep(poll_seconds)


@contextlib.asynccontextmanager
async def gpu_lease(
    resource: str,
    holder: str,
    *,
    wait_seconds: Optional[int] = None,
    ttl_seconds: Optional[int] = None,
    on_waiting: Optional[Callable[[], Union[None, Awaitable[None]]]] = None,
):
    """Acquire, heartbeat in the background, always release."""
    cap = capacity_profile()
    ttl = int(ttl_seconds if ttl_seconds is not None else cap.gpu_lease_ttl_seconds)
    await acquire_with_wait(
        resource,
        holder,
        wait_seconds=wait_seconds,
        ttl_seconds=ttl,
        on_waiting=on_waiting,
    )

    async def _renew():
        interval = max(10.0, ttl / 3)
        while True:
            await asyncio.sleep(interval)
            if not heartbeat(resource, holder, ttl_seconds=ttl):
                logger.warning(
                    "GPU lease lost resource=%s holder=%s — another worker took it",
                    resource,
                    holder,
                )
                return

    renew_task = asyncio.create_task(_renew())
    try:
        yield
    finally:
        renew_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await renew_task
        release(resource, holder)
        logger.info("GPU lease released resource=%s holder=%s", resource, holder)


def gpu_snapshot() -> dict:
    """Best-effort nvidia-smi reading for /health/capacity. Never raises."""
    import subprocess

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
            text=True,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)[:200]}

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_pct": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                }
            )
        except ValueError:
            continue
    return {"available": True, "gpus": gpus}


def lease_status() -> dict:
    snap = snapshot(RESOURCE_DOCLING)
    return {
        RESOURCE_DOCLING: (
            None
            if snap is None
            else {
                "holder": snap.holder,
                "expires_at": snap.expires_at,
                "heartbeat_at": snap.heartbeat_at,
                "live": snap.expires_at > time.time(),
            }
        )
    }
