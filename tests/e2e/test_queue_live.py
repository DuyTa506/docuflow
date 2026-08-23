"""Live queue E2E against a running DocuFlow stack on :8022.

Skipped automatically when the API is down. Uses EXTRACTED documents only —
never starts OCR. Cancels every job it creates before exiting.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
import requests

BASE = os.environ.get("DOCUFLOW_E2E_BASE", "http://localhost:8022").rstrip("/")
API = f"{BASE}/api/v2"
USER = os.environ.get("DOCUFLOW_E2E_USER", "admin")
PASSWORD = os.environ.get("DOCUFLOW_E2E_PASSWORD", "admin")


def _api_up() -> bool:
    try:
        r = requests.get(f"{BASE}/health/live", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _api_up(), reason="DocuFlow API not reachable on :8022")


@pytest.fixture(scope="module")
def token() -> str:
    r = requests.post(
        f"{API}/auth/login",
        json={"username": USER, "password": PASSWORD},
        timeout=15,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    return body.get("access_token") or body.get("token")


@pytest.fixture(scope="module")
def session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    return s


def _extracted_docs(session: requests.Session, limit: int = 20) -> list[dict[str, Any]]:
    r = session.get(f"{API}/documents", params={"page": 1, "limit": limit}, timeout=30)
    assert r.status_code == 200, r.text
    items = r.json().get("items") or []
    return [d for d in items if d.get("processing_status") == "EXTRACTED"]


def _task(session: requests.Session, task_id: str) -> dict[str, Any]:
    r = session.get(f"{API}/tasks/{task_id}", timeout=15)
    assert r.status_code == 200, r.text
    body = r.json()
    # API uses task_id; normalize for assertions.
    if "id" not in body and body.get("task_id"):
        body = {**body, "id": body["task_id"]}
    return body


def _open_translation_id(session: requests.Session, doc_id: str) -> str | None:
    r = session.get(f"{API}/documents/{doc_id}/translations", timeout=15)
    if r.status_code != 200:
        return None
    for t in r.json() or []:
        if t.get("status") in ("PENDING", "IN_PROGRESS"):
            return t.get("id")
    return None


def _cancel_translate(session: requests.Session, doc_id: str) -> None:
    tid = _open_translation_id(session, doc_id)
    if tid:
        session.delete(f"{API}/documents/{doc_id}/translations/{tid}", timeout=30)


def _cancel_digest(session: requests.Session, doc_id: str) -> None:
    session.delete(f"{API}/documents/{doc_id}/analysis", timeout=30)


def test_burst_translations_queue_without_429(session: requests.Session):
    docs = _extracted_docs(session)
    if len(docs) < 4:
        pytest.skip("Need ≥4 EXTRACTED documents for burst queue test")

    created: list[tuple[str, str]] = []  # (doc_id, task_id)
    try:
        statuses: list[int] = []
        for doc in docs[:4]:
            r = session.post(
                f"{API}/documents/{doc['id']}/translations",
                json={"target_language": "vi", "domain": "general"},
                timeout=30,
            )
            statuses.append(r.status_code)
            assert r.status_code == 200, r.text
            body = r.json()
            created.append((doc["id"], body["task_id"]))

        assert 429 not in statuses

        # Give the drain a moment to start up to MAX_CONCURRENT_TRANSLATIONS.
        time.sleep(3)
        task_rows = [_task(session, tid) for _, tid in created]
        by_id = {t["id"]: t for t in task_rows}
        running = [t for t in task_rows if t.get("status") == "RUNNING"]
        pending = [t for t in task_rows if t.get("status") == "PENDING"]
        # Terminal early is fine only if we still see the queue pattern overall.
        assert len(running) + len(pending) >= 3, task_rows
        assert len(running) <= 2  # MAX_CONCURRENT_TRANSLATIONS
        assert len(pending) >= 1

        queued = next((t for t in pending), None)
        assert queued is not None
        doc_for_queued = next(d for d, tid in created if tid == queued["id"])
        tid = _open_translation_id(session, doc_for_queued)
        assert tid, "queued translation row missing"
        cr = session.delete(f"{API}/documents/{doc_for_queued}/translations/{tid}", timeout=30)
        assert cr.status_code == 200, cr.text
        time.sleep(1)
        after = _task(session, queued["id"])
        assert after.get("status") == "CANCELLED", after
        assert by_id  # keep lint quiet if unused in some runs
    finally:
        for doc_id, _ in created:
            _cancel_translate(session, doc_id)


def test_burst_digest_and_heavy_stage_queue(session: requests.Session):
    docs = _extracted_docs(session)
    if len(docs) < 3:
        pytest.skip("Need ≥3 EXTRACTED documents for digest queue test")

    created: list[str] = []
    stage_doc = None
    try:
        for doc in docs[:3]:
            r = session.post(f"{API}/documents/{doc['id']}/analysis", json={}, timeout=30)
            assert r.status_code == 200, r.text
            assert r.status_code != 429
            created.append(doc["id"])

        time.sleep(2)
        # While digests occupy the slot, a heavy stage rerun must queue (200), not 429.
        stage_doc = docs[0]["id"]
        sr = session.post(f"{API}/documents/{stage_doc}/summaries", json={}, timeout=30)
        assert sr.status_code == 200, sr.text
        assert sr.status_code != 429
        stid = sr.json().get("task_id")
        if stid:
            time.sleep(1)
            st = _task(session, stid)
            assert st.get("status") in ("PENDING", "RUNNING", "COMPLETED", "CANCELLED", "FAILED")
    finally:
        for doc_id in created:
            _cancel_digest(session, doc_id)
        if stage_doc:
            # Best-effort: cancelling digest frees related stage capacity.
            _cancel_digest(session, stage_doc)
