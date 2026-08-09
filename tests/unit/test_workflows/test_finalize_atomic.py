"""Finalize must persist pipeline_state=DONE and the parent Task's
COMPLETED+result in ONE transaction — the old two-commit finalize could crash
between them, leaving the document DONE but the task forever RUNNING.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.db_models import Base, Document, Task
from services.pipeline.mirror import update_pipeline_mirror


def _make_manager(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    class FakeManager:
        @contextmanager
        def session(self):
            s = Session()
            try:
                yield s
                s.commit()
            finally:
                s.close()

    monkeypatch.setattr("services.pipeline.mirror.get_db_manager", lambda: FakeManager())
    return Session


def test_mirror_done_sets_task_result_in_same_call(monkeypatch):
    Session = _make_manager(monkeypatch)
    with Session() as s:
        s.add(Document(id="DOC_F", title="t", original_filename="t.pdf", total_pages=1))
        s.add(Task(id="TASK_F", document_id="DOC_F", task_type="DIGEST_PIPELINE", status="RUNNING"))
        s.commit()

    report = {"ok": True, "warnings": []}
    update_pipeline_mirror(
        "DOC_F",
        state="DONE",
        stage="FINALIZE",
        stage_progress=100,
        message="Pipeline completed",
        parent_task_id="TASK_F",
        quality_report=report,
        task_result=report,
    )

    with Session() as s:
        doc = s.query(Document).filter(Document.id == "DOC_F").first()
        task = s.query(Task).filter(Task.id == "TASK_F").first()
        assert doc.pipeline_state == "DONE"
        assert task.status == "COMPLETED"
        assert task.result == report
