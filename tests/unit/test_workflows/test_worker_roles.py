"""Extraction must be separable into its own process, since it loads Docling.

The four queues were already split but still shared one process, so the process
serving digest was the very same one that had loaded Docling's three GPU models
during extraction. `release_cached_gpu_memory()` gives back the empty part of the
pool; splitting the process is the only way the digest side **never** loads
Docling at all.

`role` selects a set of queues and nothing more — per-queue configuration is
unchanged, so running everything together (`all`) behaves exactly as before and
no deployment breaks on upgrade.
"""

import pytest

from config.settings import settings
from workers.temporal_worker import ROLES, worker_configs


def _queues(role):
    return [c["task_queue"] for c in worker_configs(role)]


class TestRoles:
    def test_all_is_the_default_and_covers_every_queue(self):
        assert _queues("all") == _queues(None)
        assert set(_queues("all")) == {
            settings.temporal_task_queue,
            settings.temporal_translation_task_queue,
            settings.temporal_extraction_task_queue,
            settings.temporal_stage_task_queue,
        }

    def test_extraction_role_serves_only_the_extraction_queue(self):
        assert _queues("extraction") == [settings.temporal_extraction_task_queue]

    def test_pipeline_role_excludes_extraction(self):
        queues = _queues("pipeline")

        assert settings.temporal_extraction_task_queue not in queues
        assert settings.temporal_task_queue in queues

    def test_the_two_split_roles_partition_all(self):
        """Two processes must serve exactly the same queue set as one."""
        assert sorted(_queues("pipeline") + _queues("extraction")) == sorted(_queues("all"))

    def test_unknown_role_is_rejected_loudly(self):
        with pytest.raises(ValueError) as exc:
            worker_configs("nonsense")

        assert "nonsense" in str(exc.value)

    def test_roles_is_the_published_list(self):
        assert set(ROLES) == {"all", "pipeline", "extraction"}


class TestConfigsAreUnchanged:
    def test_extraction_config_is_identical_in_both_roles(self):
        """Splitting the process must not change a queue's own behaviour."""
        alone = next(
            c
            for c in worker_configs("extraction")
            if c["task_queue"] == settings.temporal_extraction_task_queue
        )
        together = next(
            c
            for c in worker_configs("all")
            if c["task_queue"] == settings.temporal_extraction_task_queue
        )

        assert alone == together
