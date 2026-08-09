"""Standalone single-stage reruns must get the same durability as digest stages.

Before this, a "rerun summary" on a 700-page book ran as a bare
asyncio.create_task inside uvicorn: no timeout, no heartbeat, no retry, no
resume, and killed by any API restart. The work is identical to the digest
stage — only the dispatch differed.
"""

from datetime import timedelta

import pytest

from services.stage_dispatch import (
    LONG_STAGES,
    STAGE_RUNNERS,
    stage_policy,
)


class TestStageRegistry:
    def test_covers_every_reruntable_stage(self):
        """Every task_type a service submits standalone must have a runner —
        a missing entry means the workflow starts and immediately fails."""
        expected = {
            "BUILD_TREE",
            "BIBLIOGRAPHIC",
            "KEYWORDS",
            "RESEARCH_DIRECTIONS",
            "USAGE_SCOPE",
            "HIERARCHICAL_SUMMARIZE",
            "MAIN_CONTENT",
        }
        assert expected == set(STAGE_RUNNERS)

    def test_runners_accept_document_id_and_task_id(self):
        """Progress must land on the rerun's own Task row, so every runner has
        to take task_id — not just document_id."""
        import inspect

        for stage, runner in STAGE_RUNNERS.items():
            params = list(inspect.signature(runner).parameters)
            assert params[:2] == ["document_id", "task_id"], stage


class TestStagePolicy:
    def test_long_stages_get_hours_not_minutes(self):
        """Observed ~9-10h on an 816-page doc — a 30min cap would kill it."""
        for stage in ("HIERARCHICAL_SUMMARIZE", "MAIN_CONTENT"):
            policy = stage_policy(stage)
            assert policy.start_to_close >= timedelta(hours=12)
            assert policy.retry.maximum_attempts >= 6

    def test_short_stages_stay_bounded(self):
        policy = stage_policy("KEYWORDS")
        assert policy.start_to_close <= timedelta(minutes=30)

    def test_every_stage_heartbeats_well_inside_its_timeout(self):
        """heartbeat_timeout must leave room for the 20s ping interval, else a
        healthy long stage gets force-failed."""
        from workflows.activities._common import _HEARTBEAT_INTERVAL_SECONDS

        for stage in STAGE_RUNNERS:
            policy = stage_policy(stage)
            assert policy.heartbeat is not None, stage
            assert policy.heartbeat.total_seconds() >= _HEARTBEAT_INTERVAL_SECONDS * 3, stage

    def test_long_stages_are_the_expensive_ones(self):
        assert LONG_STAGES == {"BUILD_TREE", "HIERARCHICAL_SUMMARIZE", "MAIN_CONTENT"}

    def test_unknown_stage_rejected(self):
        with pytest.raises(KeyError):
            stage_policy("NOT_A_STAGE")


class TestOrphanSweepExcludesTemporalStages:
    def test_stage_types_excluded_when_flag_on(self, monkeypatch):
        """The API restart sweep must not FAIL rows whose work lives in the
        Temporal worker — that process survives an API restart."""
        from config.settings import settings
        from services.task_manager import temporal_owned_task_types

        monkeypatch.setattr(settings, "stage_rerun_use_temporal", True)
        owned = temporal_owned_task_types()
        for stage in STAGE_RUNNERS:
            assert stage in owned

    def test_stage_types_swept_when_flag_off(self, monkeypatch):
        """Legacy in-process path: those rows really are orphaned by a restart."""
        from config.settings import settings
        from services.task_manager import temporal_owned_task_types

        monkeypatch.setattr(settings, "stage_rerun_use_temporal", False)
        owned = temporal_owned_task_types()
        for stage in STAGE_RUNNERS:
            assert stage not in owned


class TestWorkerRegistration:
    def test_stage_activity_registered(self):
        from workers.temporal_worker import STAGE_ACTIVITIES
        from workflows.activities.stage_rerun_activities import run_stage_activity

        assert run_stage_activity in STAGE_ACTIVITIES

    def test_stage_workflow_registered_on_its_own_queue(self):
        """A multi-hour rerun must not starve digests sharing one queue."""
        from config.settings import settings
        from workers.temporal_worker import _stage_worker_config
        from workflows.stage_rerun_workflow import StageRerunWorkflow

        cfg = _stage_worker_config()
        assert StageRerunWorkflow in cfg["workflows"]
        assert cfg["task_queue"] == settings.temporal_stage_task_queue
        assert cfg["task_queue"] != settings.temporal_task_queue


class TestStallDetectionWiring:
    def test_only_progress_reporting_stages_are_stall_checked(self):
        """BUILD_TREE reports no progress, so a stall check would fail every
        healthy run of it."""
        assert stage_policy("BUILD_TREE").stall_timeout is None
        assert stage_policy("HIERARCHICAL_SUMMARIZE").stall_timeout is not None
        assert stage_policy("MAIN_CONTENT").stall_timeout is not None

    def test_stall_timeout_is_well_inside_the_hard_timeout(self):
        """The point is failing in minutes instead of burning all 12h."""
        for stage in ("HIERARCHICAL_SUMMARIZE", "MAIN_CONTENT"):
            policy = stage_policy(stage)
            assert policy.stall_timeout < policy.start_to_close / 4

    def test_stall_timeout_exceeds_heartbeat_timeout(self):
        """Otherwise the heartbeat_timeout fires first and the stall check
        never gets a chance to distinguish stalled from merely slow."""
        for stage in ("HIERARCHICAL_SUMMARIZE", "MAIN_CONTENT"):
            policy = stage_policy(stage)
            assert policy.stall_timeout > policy.heartbeat
