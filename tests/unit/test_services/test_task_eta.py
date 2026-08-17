from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from data.db_models import Base, Document, Task, TaskEtaProfile
from services.eta.digest import remaining_profile_seconds
from services.eta.ocr import dimensions as ocr_dimensions
from services.eta.profiles import get_profile, record_observation
from services.eta.translation import dimensions as translation_dimensions
from services.task_manager import TaskManager


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(Document(id="DOC_ETA", title="ETA", original_filename="eta.pdf", total_pages=10))
    session.commit()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def eta_settings(monkeypatch):
    monkeypatch.setattr(settings, "eta_enabled", True)
    monkeypatch.setattr(settings, "eta_shadow_mode", False)
    monkeypatch.setattr(settings, "eta_public_profile_keys", "*")
    monkeypatch.setattr(settings, "eta_profile_min_samples", 1)
    monkeypatch.setattr(settings, "eta_live_sample_threshold", 3)
    monkeypatch.setattr(settings, "eta_stall_min_seconds", 10)
    monkeypatch.setattr(settings, "eta_stall_p90_multiplier", 2.0)


def add_task(db, task_id="EXTRACT_ETA", task_type="EXTRACT"):
    task = Task(
        id=task_id,
        document_id="DOC_ETA",
        task_type=task_type,
        status="PENDING",
        progress=0,
    )
    db.add(task)
    db.commit()
    return task


def add_profile(db, pipeline, mode_stage, feature_bucket, p50=10, p90=20):
    db.add(
        TaskEtaProfile(
            pipeline=pipeline,
            mode_stage=mode_stage,
            feature_bucket=feature_bucket,
            rate_p50=p50,
            rate_p90=p90,
            sample_count=20,
        )
    )
    db.commit()


@pytest.mark.parametrize("mode", ["pdf_text", "pdf_scan", "pdf_hybrid", "docx", "image"])
def test_ocr_modes_have_independent_profiles(mode):
    assert ocr_dimensions({"mode": mode, "phase": "active", "units_total": 10}) == (
        "extract",
        mode,
        "small",
    )


@pytest.mark.parametrize(
    "mode",
    ["docx_inplace", "pdf_overlay", "block_based", "element_based", "tree", "flat"],
)
def test_translation_modes_have_independent_profiles(mode):
    assert translation_dimensions(
        {
            "mode": mode,
            "phase": "active",
            "units_total": 50,
            "target_language": "vi",
        }
    ) == ("translate", mode, "lang=vi;size=medium")


def test_waiting_has_no_start_or_numeric_eta(db):
    add_task(db)
    TaskManager.update_progress(
        db,
        "EXTRACT_ETA",
        0,
        "wait",
        {
            "version": 1,
            "pipeline": "extract",
            "phase": "waiting_upstream",
            "mode": "pdf_scan",
            "attempt": 1,
        },
    )
    task = db.get(Task, "EXTRACT_ETA")
    assert task.status == "RUNNING"
    assert task.started_at is None
    assert task.eta["state"] == "waiting_upstream"
    assert task.eta["low_seconds"] is None


def test_ocr_profile_eta_monotonic_units_stall_and_terminal(db):
    add_task(db)
    add_profile(db, "extract", "pdf_scan", "small")
    start = datetime(2026, 8, 17, 4, 0, 0)
    meta = {
        "version": 1,
        "pipeline": "extract",
        "phase": "active",
        "mode": "pdf_scan",
        "unit_kind": "page",
        "units_done": 0,
        "units_total": 10,
        "attempt": 1,
    }
    assert TaskManager.update_progress(db, "EXTRACT_ETA", 0, "active", meta, now=start)
    task = db.get(Task, "EXTRACT_ETA")
    assert task.started_at == start
    assert task.eta["state"] == "active"
    assert task.eta["low_seconds"] == 100
    assert task.eta["high_seconds"] == 200

    moved = {**meta, "units_done": 2}
    assert TaskManager.update_progress(
        db, "EXTRACT_ETA", 20, "page", moved, now=start + timedelta(seconds=20)
    )
    assert not TaskManager.update_progress(
        db, "EXTRACT_ETA", 10, "late", {**meta, "units_done": 1}, now=start + timedelta(seconds=21)
    )
    assert db.get(Task, "EXTRACT_ETA").progress == 20

    TaskManager.refresh_eta_state(
        db,
        db.get(Task, "EXTRACT_ETA"),
        now=start + timedelta(seconds=61),
    )
    assert db.get(Task, "EXTRACT_ETA").eta["state"] == "stalled"

    assert TaskManager.mark_terminal(
        db,
        "EXTRACT_ETA",
        status="COMPLETED",
        now=start + timedelta(seconds=70),
    )
    task = db.get(Task, "EXTRACT_ETA")
    assert task.progress == 100
    assert task.completed_at == start + timedelta(seconds=70)
    assert task.eta["state"] == "terminal"
    assert not TaskManager.update_progress(db, task.id, 99, "stale", moved)


def test_translation_uses_mode_language_and_size_profile(db):
    add_task(db, "TRANSLATE_ETA", "TRANSLATE")
    add_profile(db, "translate", "flat", "lang=vi;size=medium", p50=5, p90=8)
    TaskManager.update_progress(
        db,
        "TRANSLATE_ETA",
        10,
        "chunk",
        {
            "version": 1,
            "pipeline": "translate",
            "phase": "active",
            "mode": "flat",
            "unit_kind": "chunk",
            "units_done": 10,
            "units_total": 50,
            "attempt": 1,
            "target_language": "vi",
        },
        now=datetime(2026, 8, 17, 4, 0, 0),
    )
    task = db.get(Task, "TRANSLATE_ETA")
    assert task.eta["low_seconds"] == 200
    assert task.eta_estimator_state["profile_key"] == "translate|flat|lang=vi;size=medium"


def test_shadow_mode_withholds_numeric_range_but_keeps_validation_prediction(db, monkeypatch):
    monkeypatch.setattr(settings, "eta_shadow_mode", True)
    add_task(db)
    add_profile(db, "extract", "pdf_scan", "small")
    TaskManager.update_progress(
        db,
        "EXTRACT_ETA",
        0,
        "active",
        {
            "version": 1,
            "pipeline": "extract",
            "phase": "active",
            "mode": "pdf_scan",
            "unit_kind": "page",
            "units_done": 0,
            "units_total": 10,
            "attempt": 1,
        },
    )
    task = db.get(Task, "EXTRACT_ETA")
    assert task.eta["low_seconds"] is None
    assert task.eta_estimator_state["shadow_prediction"]["low_seconds"] == 100


def test_unapproved_profile_key_stays_hidden_when_shadow_is_disabled(db, monkeypatch):
    monkeypatch.setattr(settings, "eta_public_profile_keys", "")
    add_task(db)
    add_profile(db, "extract", "pdf_scan", "small")
    TaskManager.update_progress(
        db,
        "EXTRACT_ETA",
        0,
        "active",
        {
            "version": 1,
            "pipeline": "extract",
            "phase": "active",
            "mode": "pdf_scan",
            "unit_kind": "page",
            "units_done": 0,
            "units_total": 10,
            "attempt": 1,
        },
    )
    task = db.get(Task, "EXTRACT_ETA")
    assert task.eta["low_seconds"] is None
    assert task.eta_estimator_state["profile_key"] == "extract|pdf_scan|small"


def test_raw_percent_and_human_message_never_create_eta(db):
    add_task(db)
    TaskManager.update_progress(
        db,
        "EXTRACT_ETA",
        99,
        "Page 99/100, maybe one minute left",
        {
            "version": 1,
            "pipeline": "extract",
            "phase": "active",
            "mode": "unsupported",
            "attempt": 1,
        },
    )
    task = db.get(Task, "EXTRACT_ETA")
    assert task.eta["state"] == "active"
    assert task.eta["low_seconds"] is None
    assert task.eta["high_seconds"] is None


def test_retry_resets_live_rate_state_and_keeps_checkpoint(db):
    add_task(db)
    start = datetime(2026, 8, 17, 4, 0, 0)
    base = {
        "version": 1,
        "pipeline": "extract",
        "phase": "active",
        "mode": "pdf_hybrid",
        "unit_kind": "page",
        "units_total": 10,
        "attempt": 1,
    }
    TaskManager.update_progress(db, "EXTRACT_ETA", 20, "a", {**base, "units_done": 2}, now=start)
    TaskManager.update_progress(
        db, "EXTRACT_ETA", 30, "b", {**base, "units_done": 3}, now=start + timedelta(seconds=10)
    )
    assert db.get(Task, "EXTRACT_ETA").eta_estimator_state["sample_count"] == 1

    retry = {**base, "attempt": 2, "units_done": 3, "checkpoint_units": 3}
    TaskManager.update_progress(
        db, "EXTRACT_ETA", 30, "retry", retry, now=start + timedelta(minutes=5)
    )
    state = db.get(Task, "EXTRACT_ETA").eta_estimator_state
    assert state["attempt"] == 2
    assert state["sample_count"] == 0
    assert db.get(Task, "EXTRACT_ETA").progress_meta["checkpoint_units"] == 3


def test_profile_refresh_winsorizes_completion_outlier(db):
    add_task(db)
    for index in range(20):
        record_observation(
            db,
            task_id="EXTRACT_ETA",
            segment_key=f"segment-{index}",
            pipeline="extract",
            mode_stage="pdf_scan",
            feature_bucket="small",
            units=1,
            duration_seconds=10 if index < 19 else 10_000,
            success=True,
        )
    db.commit()
    profile = get_profile(db, "extract", "pdf_scan", "small")
    assert profile is not None
    assert profile.sample_count == 20
    assert profile.rate_p50 == pytest.approx(10)
    assert profile.rate_p90 < 100


def test_digest_parallel_groups_use_max_not_sum():
    durations = {
        "BUILD_TREE": (10, 20),
        "BIBLIOGRAPHIC": (30, 40),
        "KEYWORDS": (50, 60),
        "RESEARCH_DIRECTIONS": (20, 30),
        "USAGE_SCOPE": (10, 20),
        "HIERARCHICAL_SUMMARIZE": (100, 120),
        "MAIN_CONTENT": (80, 140),
        "FINALIZE": (5, 10),
    }
    remaining = remaining_profile_seconds({}, durations.get, current_stage=None)
    # build + max(group A) + max(group B) + finalize
    assert remaining == (10 + 50 + 100 + 5, 20 + 60 + 140 + 10)


def test_digest_keeps_independent_live_state_per_parallel_stage(db):
    add_task(db, "DIGEST_PIPELINE_ETA", "DIGEST_PIPELINE")
    base = {
        "version": 1,
        "pipeline": "digest",
        "phase": "active",
        "mode": "digest_pipeline",
        "unit_kind": "unit",
        "units_total": 10,
        "attempt": 1,
        "stages": {},
    }
    TaskManager.update_progress(
        db,
        "DIGEST_PIPELINE_ETA",
        20,
        "keywords",
        {**base, "stage": "KEYWORDS", "units_done": 2},
    )
    TaskManager.update_progress(
        db,
        "DIGEST_PIPELINE_ETA",
        25,
        "bibliographic",
        {**base, "stage": "BIBLIOGRAPHIC", "units_done": 1},
    )
    state = db.get(Task, "DIGEST_PIPELINE_ETA").eta_estimator_state
    assert set(state["stage_states"]) == {"KEYWORDS", "BIBLIOGRAPHIC"}
    assert state["stage_states"]["KEYWORDS"]["last_units"] == 2
