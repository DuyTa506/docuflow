"""Quality report tests."""
from services.pipeline.quality import build_quality_report


def test_build_quality_report_missing_doc(monkeypatch):
    def fake_assemble(self, db, document_id):
        from services.digest_service import DigestResult

        return DigestResult(
            document_id=document_id,
            title="T",
            source_language="en",
            original_filename="a.pdf",
            missing=["abstract", "keywords"],
        )

    monkeypatch.setattr(
        "services.pipeline.quality.DigestService.assemble",
        fake_assemble,
    )

    class FakeQuery:
        def filter(self, *a, **k):
            return self

        def order_by(self, *a, **k):
            return self

        def first(self):
            return None

    class FakeSession:
        def query(self, *a):
            return FakeQuery()

    class FakeDbManager:
        def session(self):
            from contextlib import contextmanager

            @contextmanager
            def cm():
                yield FakeSession()

            return cm()

    monkeypatch.setattr(
        "services.pipeline.quality.get_db_manager",
        lambda: FakeDbManager(),
    )

    report = build_quality_report("DOC_TEST")
    assert report["ok"] is False
    assert "abstract" in report["missing"]
    assert any("TreeIndex" in w for w in report["warnings"])
