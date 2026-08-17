"""Single-host capacity profile shared by API submit, Temporal workers, and GPU leases.

Every service used to pick its own concurrency. On one GPU that oversubscribed
llama.cpp, let four digest stages plus OCR contend for VRAM, and ignored
``MAX_CONCURRENT_PIPELINES``. This module is the one place those caps live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

from config.settings import settings

T = TypeVar("T")

SLOT_DIGEST = "DIGEST_PIPELINE"
SLOT_EXTRACT = "EXTRACT"
SLOT_TRANSLATE = "TRANSLATE"
SLOT_STAGE = "STAGE_RERUN"

HEAVY_STAGE_TYPES = frozenset({"HIERARCHICAL_SUMMARIZE", "MAIN_CONTENT", "BUILD_TREE"})


@dataclass(frozen=True)
class CapacityProfile:
    max_digest_pipelines: int
    max_extractions: int
    max_translations: int
    max_jobs_per_user: int
    digest_group_a_parallelism: int
    digest_group_b_parallel: bool
    gpu_docling_slots: int
    gpu_lease_ttl_seconds: int
    gpu_lease_wait_seconds: int


def capacity_profile() -> CapacityProfile:
    return CapacityProfile(
        max_digest_pipelines=max(1, settings.max_concurrent_pipelines),
        max_extractions=max(1, settings.extraction_max_concurrent),
        max_translations=max(1, settings.max_concurrent_translations),
        max_jobs_per_user=max(1, settings.max_concurrent_jobs_per_user),
        digest_group_a_parallelism=max(1, settings.digest_group_a_parallelism),
        digest_group_b_parallel=bool(settings.digest_group_b_parallel),
        gpu_docling_slots=max(1, settings.gpu_docling_slots),
        gpu_lease_ttl_seconds=max(30, settings.gpu_lease_ttl_seconds),
        gpu_lease_wait_seconds=max(5, settings.gpu_lease_wait_seconds),
    )


def batched(items: Sequence[T], size: int) -> list[Sequence[T]]:
    """Split ``items`` into consecutive batches of at most ``size``."""
    n = max(int(size), 1)
    return [items[i : i + n] for i in range(0, len(items), n)]
