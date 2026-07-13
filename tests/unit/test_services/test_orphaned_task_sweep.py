"""On API startup, Task rows left PENDING/RUNNING by a crashed process must be
failed proactively — in-process asyncio tasks die with the process, and the old
lazy cleanup (only on next submit for the same document) fixed only the Task
row, leaving Translation rows IN_PROGRESS forever.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.db_models import Base, Document, Task, Translation
from services.task_manager import fail_orphaned_tasks


@pytest.fixture
def isolated_db():
    """Own in-memory engine — the shared session-scoped fixture leaks
    PENDING/RUNNING rows from other tests into the sweep count."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _seed(db):
    db.add(Document(id="DOC_ORPH", title="t", original_filename="t.pdf", total_pages=1))
    db.add(
        Task(
            id="TRANSLATE_900",
            document_id="DOC_ORPH",
            task_type="TRANSLATE",
            status="RUNNING",
        )
    )
    db.add(
        Task(
            id="EXTRACT_901",
            document_id="DOC_ORPH",
            task_type="EXTRACT",
            status="PENDING",
        )
    )
    db.add(
        Task(
            id="DIGEST_PIPELINE_902",
            document_id="DOC_ORPH",
            task_type="DIGEST_PIPELINE",
            status="RUNNING",
        )
    )
    db.add(
        Translation(
            id="TRN_900",
            document_id="DOC_ORPH",
            target_language="vi",
            status="IN_PROGRESS",
        )
    )
    db.commit()


def test_orphaned_tasks_and_translations_failed_on_sweep(isolated_db):
    test_db_session = isolated_db
    _seed(test_db_session)

    count = fail_orphaned_tasks(test_db_session)

    translate = test_db_session.query(Task).filter(Task.id == "TRANSLATE_900").first()
    extract = test_db_session.query(Task).filter(Task.id == "EXTRACT_901").first()
    digest = test_db_session.query(Task).filter(Task.id == "DIGEST_PIPELINE_902").first()
    trn = test_db_session.query(Translation).filter(Translation.id == "TRN_900").first()

    assert translate.status == "FAILED"
    assert "restart" in (translate.error or "").lower()
    assert extract.status == "FAILED"
    # Temporal owns DIGEST_PIPELINE tasks — the workflow survives an API
    # restart, so the sweep must NOT touch them.
    assert digest.status == "RUNNING"
    assert trn.status == "FAILED"
    assert count == 3
