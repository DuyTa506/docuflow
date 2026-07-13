"""Quality report should surface degraded (fallback-text) chapters as a warning."""

from contextlib import contextmanager
from unittest.mock import MagicMock

from services.pipeline.quality import build_quality_report


def test_build_quality_report_warns_on_degraded_chapters(monkeypatch):
    def fake_assemble(self, db, document_id):
        from services.digest_service import DigestResult

        return DigestResult(
            document_id=document_id,
            title="T",
            source_language="en",
            original_filename="a.pdf",
            missing=[],
        )

    monkeypatch.setattr(
        "services.pipeline.quality.DigestService.assemble",
        fake_assemble,
    )

    fake_main_content = MagicMock()
    fake_main_content.details = {
        "chapters": [{"number": 1, "title_vi": "x", "content": "y"}],
        "degraded_chapters": 2,
    }

    class FakeQuery:
        def __init__(self, model):
            self._model = model

        def filter(self, *a, **k):
            return self

        def order_by(self, *a, **k):
            return self

        def first(self):
            from data.db_models import MainContent, TreeIndex

            if self._model is TreeIndex:
                return object()
            if self._model is MainContent:
                return fake_main_content
            return None

    class FakeSession:
        def query(self, model):
            return FakeQuery(model)

    class FakeDbManager:
        def session(self):
            @contextmanager
            def cm():
                yield FakeSession()

            return cm()

    monkeypatch.setattr(
        "services.pipeline.quality.get_db_manager",
        lambda: FakeDbManager(),
    )

    report = build_quality_report("DOC_TEST")
    assert any("2 chương" in w for w in report["warnings"])


def test_build_quality_report_warns_on_raw_passthrough_chapters(monkeypatch):
    """Raw-passthrough (LLM never called, content too short) is a distinct,
    previously-invisible quality signal from `degraded` (LLM call failed) --
    confirmed real on a 761-page document where ~28% of chapters still hit
    this path even after the children-text-aggregation fix."""

    def fake_assemble(self, db, document_id):
        from services.digest_service import DigestResult

        return DigestResult(
            document_id=document_id,
            title="T",
            source_language="en",
            original_filename="a.pdf",
            missing=[],
        )

    monkeypatch.setattr(
        "services.pipeline.quality.DigestService.assemble",
        fake_assemble,
    )

    fake_main_content = MagicMock()
    fake_main_content.details = {
        "chapters": [{"number": 1, "title_vi": "x", "content": "y"}],
        "degraded_chapters": 0,
        "raw_passthrough_chapters": 5,
    }

    class FakeQuery:
        def __init__(self, model):
            self._model = model

        def filter(self, *a, **k):
            return self

        def order_by(self, *a, **k):
            return self

        def first(self):
            from data.db_models import MainContent, TreeIndex

            if self._model is TreeIndex:
                return object()
            if self._model is MainContent:
                return fake_main_content
            return None

    class FakeSession:
        def query(self, model):
            return FakeQuery(model)

    class FakeDbManager:
        def session(self):
            @contextmanager
            def cm():
                yield FakeSession()

            return cm()

    monkeypatch.setattr(
        "services.pipeline.quality.get_db_manager",
        lambda: FakeDbManager(),
    )

    report = build_quality_report("DOC_TEST")
    assert report["raw_passthrough_chapters"] == 5
    assert any("5 chương" in w for w in report["warnings"])
