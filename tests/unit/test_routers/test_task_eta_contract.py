from datetime import datetime

from data.db_models import Task
from services.task_manager import TaskManager
from serving.routers.events_router import _sanitize_event


def test_rest_and_sse_share_public_eta_progress_shape():
    task = Task(
        id="EXTRACT_1",
        document_id="DOC_1",
        task_type="EXTRACT",
        status="RUNNING",
        progress=25,
        message="page",
        started_at=datetime(2026, 8, 17, 4, 0, 0),
        progress_meta={
            "version": 1,
            "pipeline": "extract",
            "phase": "active",
            "mode": "pdf_scan",
            "unit_kind": "page",
            "units_done": 1,
            "units_total": 4,
            "attempt": 1,
        },
        eta={
            "state": "active",
            "low_seconds": 60,
            "high_seconds": 120,
            "confidence": 0.8,
            "estimated_finish_at": "2026-08-17T04:02:00Z",
            "calculated_at": "2026-08-17T04:00:00Z",
        },
        eta_estimator_state={"private": "must-not-leak"},
    )
    rest = TaskManager.serialize_task(task, include_result=False)
    sse = _sanitize_event(rest)
    assert sse["progress_meta"] == rest["progress_meta"]
    assert sse["eta"] == rest["eta"]
    assert "eta_estimator_state" not in rest
    assert "eta_estimator_state" not in sse


def test_sse_sanitizes_unknown_nested_fields():
    payload = _sanitize_event(
        {
            "task_id": "T",
            "document_id": "D",
            "task_type": "EXTRACT",
            "status": "RUNNING",
            "progress": 1,
            "message": "x" * 1000,
            "progress_meta": {
                "version": 1,
                "pipeline": "extract",
                "phase": "active",
                "secret": "drop",
            },
            "eta": {"state": "active", "confidence": 0.1, "secret": "drop"},
        }
    )
    assert len(payload["message"]) == 500
    assert "secret" not in payload["progress_meta"]
    assert "secret" not in payload["eta"]
