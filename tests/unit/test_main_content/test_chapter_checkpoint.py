from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.db_models import Base, Document, MainContent
from services import main_content_service as svc
from services.main_content_service import (
    _chapter_resume_key,
    _load_checkpoint_chapters,
    _persist_checkpoint_chapters,
)


def test_resume_key_prefers_node_id():
    assert _chapter_resume_key({"node_id": "n1", "title": "Intro"}) == "n1"
    assert _chapter_resume_key({"title": "Intro"}) == "Intro"


def test_persist_and_load_checkpoint_chapters(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    class _Mgr:
        @contextmanager
        def session(self):
            db = Session()
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

    monkeypatch.setattr(svc, "get_db_manager", lambda: _Mgr())

    with Session() as db:
        db.add(
            Document(
                id="DOC_1",
                title="Doc",
                original_filename="doc.pdf",
                format="pdf",
                file_path="documents/DOC_1/original/doc.pdf",
                file_type="pdf",
                processing_status="EXTRACTED",
            )
        )
        db.add(
            MainContent(
                id="MC_1",
                document_id="DOC_1",
                details={},
                status="IN_PROGRESS",
            )
        )
        db.commit()

    _persist_checkpoint_chapters(
        "MC_1",
        [{"number": 1, "resume_key": "n1", "title_vi": "Mở đầu", "content": "…"}],
    )
    loaded = _load_checkpoint_chapters("MC_1")
    assert loaded["n1"]["title_vi"] == "Mở đầu"
