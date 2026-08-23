"""Production-hardening unit checks for cancel, OCR timeout, digest finalize."""

from config.settings import settings
from services.task_manager import TaskManager
from workflows.timeouts import HEARTBEAT, LONG_RUN


def test_fail_latest_open_marks_cancelled():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from data.db_models import Base, Document, Task
    from data.id_generator import IdGenerator

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    IdGenerator.seed_sequences(db)
    db.add(Document(id="DOC_C", title="t", original_filename="t.pdf", total_pages=1))
    db.add(
        Task(
            id="TRANSLATE_1",
            document_id="DOC_C",
            task_type="TRANSLATE",
            status="PENDING",
            progress=0,
        )
    )
    db.commit()

    task = TaskManager.fail_latest_open(db, "DOC_C", "TRANSLATE", commit=True)
    assert task is not None
    db.refresh(task)
    assert task.status == "CANCELLED"
    assert "hủy" in (task.message or "").lower()


def test_update_progress_does_not_resurrect_cancelled():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from data.db_models import Base, Document, Task
    from data.id_generator import IdGenerator

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    IdGenerator.seed_sequences(db)
    db.add(Document(id="DOC_C", title="t", original_filename="t.pdf", total_pages=1))
    db.add(
        Task(
            id="TRANSLATE_1",
            document_id="DOC_C",
            task_type="TRANSLATE",
            status="CANCELLED",
            progress=0,
            error="Cancelled by user",
        )
    )
    db.commit()
    assert TaskManager.update_progress(db, "TRANSLATE_1", 10, "should ignore") is False
    task = db.query(Task).filter(Task.id == "TRANSLATE_1").first()
    assert task.status == "CANCELLED"
    assert task.progress == 0


def test_ocr_async_openai_sets_timeout():
    from pathlib import Path

    src = Path("services/document_service.py").read_text(encoding="utf-8")
    assert "AsyncOpenAI(" in src
    assert "timeout=settings.ai_request_timeout_seconds" in src
    assert settings.ai_request_timeout_seconds > 0


def test_digest_finalize_uses_long_run_and_heartbeat():
    import ast
    from pathlib import Path

    src = Path("workflows/digest_workflow.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # execute_activity(finalize_digest_activity, ..., start_to_close_timeout=LONG_RUN)
        if not any(
            isinstance(a, ast.Name) and a.id == "finalize_digest_activity" for a in node.args
        ):
            # keyword form args=[...]
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        stc = kw.get("start_to_close_timeout")
        hb = kw.get("heartbeat_timeout")
        if isinstance(stc, ast.Name) and stc.id == "LONG_RUN":
            assert isinstance(hb, ast.Name) and hb.id == "HEARTBEAT"
            found = True
            break
        # args=[PipelineStageInput(...), ...] form — look for Name finalize in Call
    if not found:
        # Fallback: string presence (workflow sandboxes make AST of execute_activity awkward)
        assert "finalize_digest_activity" in src
        assert "start_to_close_timeout=LONG_RUN" in src
        assert "heartbeat_timeout=HEARTBEAT" in src
    assert LONG_RUN.days >= 1
    assert HEARTBEAT.total_seconds() > 0


def test_create_stage_task_queues_long_stages():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from data.db_models import Base, Document
    from data.id_generator import IdGenerator
    from services.pipeline.admission import is_queued
    from services.stage_dispatch import create_stage_task

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    IdGenerator.seed_sequences(db)
    db.add(Document(id="DOC_S", title="t", original_filename="t.pdf", total_pages=1))
    db.commit()

    task_id = create_stage_task(db, "DOC_S", "HIERARCHICAL_SUMMARIZE", fairness_key="USR_1")
    from data.db_models import Task

    task = db.query(Task).filter(Task.id == task_id).first()
    assert is_queued(task)

    short_id = create_stage_task(db, "DOC_S", "KEYWORDS")
    short = db.query(Task).filter(Task.id == short_id).first()
    assert not is_queued(short)
