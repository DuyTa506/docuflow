"""The MSSQL sink writes a job's status and results into the library's tables.

Runs on SQLite: the sink uses only portable SQLAlchemy Core (update, then insert
when nothing matched), so the same code is exercised here and on SQL Server.
The DDL the library runs (deploy/integration/mssql_schema.sql) must declare
every column the sink writes.
"""

import re
from pathlib import Path

import pytest
import sqlalchemy as sa

from services.digest_service import (
    ChapterEntry,
    DigestResult,
    KeywordEntry,
    ResearchDirectionEntry,
    digest_to_dict,
)
from services.integration_sink import MssqlSink

ROOT = Path(__file__).resolve().parents[3]
DDL = ROOT / "deploy" / "integration" / "mssql_schema.sql"


def _job(**over):
    job = {
        "job_id": "JOB_001",
        "external_id": "LIB-2024-0042",
        "document_id": "DOC_042",
        "status": "PROCESSING",
        "tasks": ["ocr", "digest"],
        "stages": {"ocr": "COMPLETED", "digest": "RUNNING"},
        "error": None,
        "created_at": "2026-09-27T08:00:00",
        "updated_at": "2026-09-27T08:05:00",
        "completed_at": None,
        "links": {"result": "http://df/api/v2/integration/jobs/JOB_001/result"},
    }
    job.update(over)
    return job


def _result(job, chapters=2):
    return {
        "job": job,
        "document": {"id": "DOC_042", "title": "Ballistics", "source_language": "en"},
        "ocr": {"text": "Chapter 1 Interior ballistics …"},
        # The real public shape, so a field that is not a string (main_content
        # is a dict) cannot slip past the SQLite tests and fail on SQL Server.
        "digest": digest_to_dict(
            DigestResult(
                document_id="DOC_042",
                title="Ballistics",
                source_language="en",
                original_filename="b.pdf",
                bibliographic={
                    "title_display": "Ballistics (Đạn đạo)",
                    "authors": "D. Carlucci, S. Jacobson",
                    "publisher": "CRC Press",
                    "year": "2008",
                    "isbn": "978-1-4200-6618-0",
                    "doi": "",
                    "pages": "502",
                },
                abstract="Tài liệu trình bày đạn đạo.",
                chapters=[ChapterEntry(i, f"Chương {i}", f"Chapter {i}", "…") for i in range(1, chapters + 1)],
                keywords=[KeywordEntry("ballistics", 0.9, "đạn đạo (ballistics)")],
                usage_scope={"undergraduate": ["Kỹ thuật vũ khí"]},
                research_directions=[ResearchDirectionEntry("Mô phỏng", 0.8)],
            )
        ),
        "translation": {"target_language": "vi", "text": "Chương 1 Đạn đạo trong …"},
        "catalog": {"marcxml": "<record/>", "dublin_core": "<oai_dc:dc/>"},
    }


@pytest.fixture
def sink():
    engine = sa.create_engine("sqlite://")
    s = MssqlSink(engine=engine, schema=None)
    s.metadata.create_all(engine)
    return s


def _rows(sink, table):
    with sink.engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(sa.select(sink.tables[table]))]


class TestJobStatus:
    def test_status_row_is_upserted(self, sink):
        sink.write(_job())
        sink.write(_job(status="COMPLETED", stages={"ocr": "COMPLETED", "digest": "COMPLETED"},
                        completed_at="2026-09-27T09:00:00"))

        rows = _rows(sink, "DocuFlowJob")
        assert len(rows) == 1
        row = rows[0]
        assert row["JobId"] == "JOB_001"
        assert row["ExternalId"] == "LIB-2024-0042"
        assert row["Status"] == "COMPLETED"
        assert row["OcrStatus"] == "COMPLETED"
        assert row["DigestStatus"] == "COMPLETED"
        assert row["TranslateStatus"] is None
        assert row["CompletedAt"].isoformat() == "2026-09-27T09:00:00"
        assert row["ResultUrl"].endswith("/jobs/JOB_001/result")

    def test_status_only_write_leaves_no_result(self, sink):
        sink.write(_job())
        assert _rows(sink, "DocuFlowResult") == []


class TestResults:
    def test_result_lands_in_catalogue_columns(self, sink):
        job = _job(status="COMPLETED")
        sink.write(job, _result(job))

        (res,) = _rows(sink, "DocuFlowResult")
        assert res["Title"] == "Ballistics (Đạn đạo)"
        assert res["Authors"] == "D. Carlucci, S. Jacobson"
        assert res["PublishYear"] == "2008"
        assert res["ISBN"] == "978-1-4200-6618-0"
        assert res["DOI"] is None  # empty strings are not values
        assert res["Abstract"] == "Tài liệu trình bày đạn đạo."
        assert res["Keywords"] == "đạn đạo (ballistics)"
        assert res["OcrText"].startswith("Chapter 1")
        assert res["TranslationText"].startswith("Chương 1")
        assert res["MarcXml"] == "<record/>"
        assert '"Kỹ thuật vũ khí"' in res["UsageScope"]  # JSON, not escaped
        assert '"chapters"' in res["DigestJson"]

        chapters = _rows(sink, "DocuFlowChapter")
        assert [(c["Ord"], c["ChapterNo"], c["TitleVi"]) for c in chapters] == [
            (1, 1, "Chương 1"),
            (2, 2, "Chương 2"),
        ]
        (kw,) = _rows(sink, "DocuFlowKeyword")
        assert (kw["Keyword"], kw["Display"], kw["Weight"]) == ("ballistics", "đạn đạo (ballistics)", 0.9)

    def test_rewrite_replaces_children(self, sink):
        job = _job(status="COMPLETED")
        sink.write(job, _result(job, chapters=3))
        sink.write(job, _result(job, chapters=1))

        assert len(_rows(sink, "DocuFlowResult")) == 1
        assert len(_rows(sink, "DocuFlowChapter")) == 1

    def test_ocr_only_job(self, sink):
        job = _job(status="COMPLETED", tasks=["ocr"], stages={"ocr": "COMPLETED"})
        result = _result(job)
        result.update(digest=None, translation=None, catalog=None)
        sink.write(job, result)

        (res,) = _rows(sink, "DocuFlowResult")
        assert res["OcrText"].startswith("Chapter 1")
        assert res["Title"] is None and res["DigestJson"] is None
        assert _rows(sink, "DocuFlowChapter") == []


def test_ddl_declares_every_column_the_sink_writes(sink):
    ddl = DDL.read_text(encoding="utf-8")
    for name, table in sink.tables.items():
        block = re.search(rf"CREATE TABLE\s+\[?docuflow\]?\.\[?{name}\]?\s*\((.*?)\n\);", ddl, re.S)
        assert block, f"{name} missing from {DDL.name}"
        declared = set(re.findall(r"^\s*\[?(\w+)\]?\s+(?:NVARCHAR|INT|FLOAT|DATETIME2)", block.group(1), re.M))
        assert declared == {c.name for c in table.columns}, name
