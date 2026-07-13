"""Temporal-routed translation/extraction activities call update_progress()
directly (no in-process _run_wrapper to flip PENDING -> RUNNING first), so a
task sat at literal status "PENDING" for its entire run and only jumped to
COMPLETED/FAILED at the end — the FE showed a stale "PENDING" badge for hours
on long documents even though real progress was happening underneath.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.db_models import Base, Document, Task
from services.task_manager import TaskManager


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def test_update_progress_flips_pending_to_running(db):
    db.add(Document(id="DOC_UP", title="t", original_filename="t.pdf", total_pages=1))
    db.add(Task(id="TRANSLATE_1", document_id="DOC_UP", task_type="TRANSLATE", status="PENDING"))
    db.commit()

    TaskManager.update_progress(db, "TRANSLATE_1", 10, "Unit 1/100")

    task = db.query(Task).filter(Task.id == "TRANSLATE_1").first()
    assert task.status == "RUNNING"
    assert task.progress == 10


def test_update_progress_leaves_running_status_alone(db):
    db.add(Document(id="DOC_UP2", title="t", original_filename="t.pdf", total_pages=1))
    db.add(Task(id="TRANSLATE_2", document_id="DOC_UP2", task_type="TRANSLATE", status="RUNNING"))
    db.commit()

    TaskManager.update_progress(db, "TRANSLATE_2", 50, "Unit 50/100")

    task = db.query(Task).filter(Task.id == "TRANSLATE_2").first()
    assert task.status == "RUNNING"


def test_update_progress_does_not_resurrect_terminal_status(db):
    """A late/duplicate progress callback after the activity already finished
    must not un-complete or un-fail the task."""
    db.add(Document(id="DOC_UP3", title="t", original_filename="t.pdf", total_pages=1))
    db.add(Task(id="TRANSLATE_3", document_id="DOC_UP3", task_type="TRANSLATE", status="COMPLETED"))
    db.commit()

    TaskManager.update_progress(db, "TRANSLATE_3", 99, "stale callback")

    task = db.query(Task).filter(Task.id == "TRANSLATE_3").first()
    assert task.status == "COMPLETED"
