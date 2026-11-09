"""Single-host soft safety ceilings + resource lease knobs.

Job-level caps (digest / extract / translate) are **RAM / process safety**,
not engine request schedulers. Under the ceiling, HTTP submit starts Temporal
immediately; only overflow waits in Postgres. Real throughput is bounded by
``AI_MAX_CONCURRENT_REQUESTS``, vLLM ``max-num-seqs``, and Docling CPU slots.
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
    docling_slots: int
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
        docling_slots=max(1, settings.docling_slots),
        gpu_lease_ttl_seconds=max(30, settings.gpu_lease_ttl_seconds),
        # 0 = wait forever for Docling lease (activity heartbeats keep Temporal alive)
        gpu_lease_wait_seconds=max(0, settings.gpu_lease_wait_seconds),
    )


def batched(items: Sequence[T], size: int) -> list[Sequence[T]]:
    """Split ``items`` into consecutive batches of at most ``size``."""
    n = max(int(size), 1)
    return [items[i : i + n] for i in range(0, len(items), n)]
