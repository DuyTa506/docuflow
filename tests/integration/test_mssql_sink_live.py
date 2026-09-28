"""The MSSQL sink against a real SQL Server (the demo container).

  docker compose -f deploy/integration/demo/docker-compose.yml up -d
  LIBRARY_MSSQL_TEST_URL='mssql+pymssql://sa:DocuFlow#Demo2026@localhost:14330/master' \
      pytest tests/integration/test_mssql_sink_live.py --override-ini="addopts="

Skipped when the variable is unset. Checks what SQLite cannot: the shipped DDL
runs (twice — it is idempotent), and Vietnamese text survives NVARCHAR.
"""

import os
from pathlib import Path

import pytest
import sqlalchemy as sa

from services.integration_sink import MssqlSink, apply_schema

URL = os.environ.get("LIBRARY_MSSQL_TEST_URL")
DDL = Path(__file__).resolve().parents[2] / "deploy" / "integration" / "mssql_schema.sql"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL, reason="LIBRARY_MSSQL_TEST_URL not set"),
]

JOB_ID = "JOB_LIVE_TEST"


@pytest.fixture
def sink():
    s = MssqlSink(URL)
    apply_schema(s.engine, DDL.read_text(encoding="utf-8"))
    apply_schema(s.engine, DDL.read_text(encoding="utf-8"))
    yield s
    with s.engine.begin() as conn:
        for name in ("DocuFlowKeyword", "DocuFlowChapter", "DocuFlowResult", "DocuFlowJob"):
            t = s.tables[name]
            conn.execute(t.delete().where(t.c.JobId == JOB_ID))


def test_round_trip_on_sql_server(sink):
    job = {
        "job_id": JOB_ID,
        "external_id": "LIB-TEST",
        "document_id": "DOC_999",
        "status": "COMPLETED",
        "stages": {"ocr": "COMPLETED", "digest": "COMPLETED"},
        "created_at": "2026-09-27T08:00:00",
        "completed_at": "2026-09-27T09:00:00",
        "links": {"result": "http://df/result"},
    }
    result = {
        "document": {"source_language": "zh"},
        "ocr": {"text": "第一章 引言" * 5000},  # > 4000 chars: needs NVARCHAR(MAX)
        "digest": {
            "bibliographic": {"title_display": "Đạn đạo học (弹道学)", "year": "2008"},
            "abstract": "Tài liệu trình bày đạn đạo trong, ngoài và tận cùng.",
            "chapters": [{"number": 1, "title_vi": "Mở đầu", "title_original": "引言", "content": "…"}],
            "main_content": {"chapters": [{"number": 1, "title_vi": "Mở đầu"}]},  # a dict: not a column
            "keywords": [{"keyword": "弹道", "display": "đạn đạo (弹道)", "weight": 0.9}],
        },
        "catalog": {"marcxml": "<record/>"},
    }
    sink.write({**job, "status": "PROCESSING", "completed_at": None})
    sink.write(job, result)

    with sink.engine.connect() as conn:
        res = conn.execute(
            sa.select(sink.tables["DocuFlowResult"]).where(sink.tables["DocuFlowResult"].c.JobId == JOB_ID)
        ).mappings().one()
        status = conn.execute(
            sa.select(sink.tables["DocuFlowJob"].c.Status).where(sink.tables["DocuFlowJob"].c.JobId == JOB_ID)
        ).scalar_one()
        kw = conn.execute(
            sa.select(sink.tables["DocuFlowKeyword"].c.Display).where(sink.tables["DocuFlowKeyword"].c.JobId == JOB_ID)
        ).scalar_one()

    assert status == "COMPLETED"
    assert res["Title"] == "Đạn đạo học (弹道学)"
    assert res["Abstract"].startswith("Tài liệu trình bày đạn đạo")
    assert len(res["OcrText"]) == len("第一章 引言") * 5000
    assert kw == "đạn đạo (弹道)"
