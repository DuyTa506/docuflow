"""Write integration job status and results into the library's MSSQL database.

The library system gets DocuFlow's output where it already works: four tables
in its own database (schema ``docuflow``, created by
``deploy/integration/mssql_schema.sql``):

  DocuFlowJob      one row per job — status of each stage, kept current
  DocuFlowResult   one row per finished job — catalogue fields, texts, MARCXML
  DocuFlowChapter  the digest's chapters, one row each
  DocuFlowKeyword  the digest's keywords, one row each

Input is the public API's own JSON (the job view and the result document), so
what lands in MSSQL is exactly what ``GET /jobs/{id}/result`` returns. Writes use
only portable SQLAlchemy Core — update, then insert when nothing matched — so
the sink runs unchanged on SQLite in tests.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

import sqlalchemy as sa

_S = sa.Unicode
_TEXT = sa.UnicodeText


def _tables(metadata: sa.MetaData, schema: Optional[str]) -> dict:
    job = sa.Table(
        "DocuFlowJob",
        metadata,
        sa.Column("JobId", _S(32), primary_key=True),
        sa.Column("ExternalId", _S(100)),
        sa.Column("DocumentId", _S(32)),
        sa.Column("Status", _S(20), nullable=False),
        sa.Column("OcrStatus", _S(20)),
        sa.Column("DigestStatus", _S(20)),
        sa.Column("TranslateStatus", _S(20)),
        sa.Column("Error", _TEXT),
        sa.Column("ResultUrl", _S(500)),
        sa.Column("CreatedAt", sa.DateTime),
        sa.Column("UpdatedAt", sa.DateTime),
        sa.Column("CompletedAt", sa.DateTime),
        sa.Column("SyncedAt", sa.DateTime, nullable=False),
        schema=schema,
    )
    result = sa.Table(
        "DocuFlowResult",
        metadata,
        sa.Column("JobId", _S(32), primary_key=True),
        sa.Column("ExternalId", _S(100)),
        sa.Column("DocumentId", _S(32)),
        sa.Column("Title", _S(1000)),
        sa.Column("Authors", _S(1000)),
        sa.Column("Publisher", _S(500)),
        sa.Column("PublishYear", _S(20)),
        sa.Column("ISBN", _S(50)),
        sa.Column("DOI", _S(200)),
        sa.Column("Pages", _S(20)),
        sa.Column("SourceLanguage", _S(10)),
        sa.Column("Abstract", _TEXT),
        sa.Column("Keywords", _TEXT),
        sa.Column("UsageScope", _TEXT),
        sa.Column("ResearchDirections", _TEXT),
        sa.Column("MarcXml", _TEXT),
        sa.Column("DublinCore", _TEXT),
        sa.Column("DigestJson", _TEXT),
        sa.Column("OcrText", _TEXT),
        sa.Column("TranslationText", _TEXT),
        sa.Column("SyncedAt", sa.DateTime, nullable=False),
        schema=schema,
    )
    chapter = sa.Table(
        "DocuFlowChapter",
        metadata,
        sa.Column("JobId", _S(32), primary_key=True),
        sa.Column("Ord", sa.Integer, primary_key=True),
        sa.Column("ChapterNo", sa.Integer),
        sa.Column("TitleVi", _S(1000)),
        sa.Column("TitleOriginal", _S(1000)),
        sa.Column("Content", _TEXT),
        schema=schema,
    )
    keyword = sa.Table(
        "DocuFlowKeyword",
        metadata,
        sa.Column("JobId", _S(32), primary_key=True),
        sa.Column("Ord", sa.Integer, primary_key=True),
        sa.Column("Keyword", _S(500)),
        sa.Column("Display", _S(500)),
        sa.Column("Weight", sa.Float),
        schema=schema,
    )
    return {t.name: t for t in (job, result, chapter, keyword)}


def _value(v):
    """Empty strings and empty collections are absent values, stored as NULL."""
    return v if v not in ("", None, [], {}) else None


def _dt(v) -> Optional[datetime]:
    return datetime.fromisoformat(v) if isinstance(v, str) and v else v or None


def _json(v) -> Optional[str]:
    return json.dumps(v, ensure_ascii=False) if _value(v) is not None else None


class MssqlSink:
    def __init__(self, url: str = "", *, schema: Optional[str] = "docuflow", engine=None):
        self.engine = engine if engine is not None else sa.create_engine(url, pool_pre_ping=True)
        self.metadata = sa.MetaData()
        self.tables = _tables(self.metadata, schema)

    def write(self, job: dict, result: Optional[dict] = None) -> None:
        """Upsert the job row and, when given, replace the job's results."""
        now = datetime.utcnow()
        stages = job.get("stages") or {}
        job_row = {
            "ExternalId": job.get("external_id"),
            "DocumentId": job.get("document_id"),
            "Status": job["status"],
            "OcrStatus": stages.get("ocr"),
            "DigestStatus": stages.get("digest"),
            "TranslateStatus": stages.get("translate"),
            "Error": job.get("error"),
            "ResultUrl": (job.get("links") or {}).get("result"),
            "CreatedAt": _dt(job.get("created_at")),
            "UpdatedAt": _dt(job.get("updated_at")),
            "CompletedAt": _dt(job.get("completed_at")),
            "SyncedAt": now,
        }
        with self.engine.begin() as conn:
            self._upsert(conn, "DocuFlowJob", job["job_id"], job_row)
            if result is not None:
                self._write_result(conn, job, result, now)

    def _upsert(self, conn, name: str, job_id: str, row: dict) -> None:
        table = self.tables[name]
        updated = conn.execute(table.update().where(table.c.JobId == job_id).values(**row))
        if updated.rowcount == 0:
            conn.execute(table.insert().values(JobId=job_id, **row))

    def _write_result(self, conn, job: dict, result: dict, now: datetime) -> None:
        job_id = job["job_id"]
        digest = result.get("digest") or {}
        bib = digest.get("bibliographic") or {}
        catalog = result.get("catalog") or {}
        keywords = digest.get("keywords") or []
        row = {
            "ExternalId": job.get("external_id"),
            "DocumentId": job.get("document_id"),
            "Title": _value(bib.get("title_display")),
            "Authors": _value(bib.get("authors")),
            "Publisher": _value(bib.get("publisher")),
            "PublishYear": _value(bib.get("year")),
            "ISBN": _value(bib.get("isbn")),
            "DOI": _value(bib.get("doi")),
            "Pages": _value(bib.get("pages")),
            "SourceLanguage": (result.get("document") or {}).get("source_language"),
            "Abstract": _value(digest.get("abstract")),
            "Keywords": _value("; ".join(k.get("display") or k["keyword"] for k in keywords)),
            "UsageScope": _json(digest.get("usage_scope")),
            "ResearchDirections": _json(digest.get("research_directions")),
            "MarcXml": _value(catalog.get("marcxml")),
            "DublinCore": _value(catalog.get("dublin_core")),
            "DigestJson": _json(digest),
            "OcrText": _value((result.get("ocr") or {}).get("text")),
            "TranslationText": _value((result.get("translation") or {}).get("text")),
            "SyncedAt": now,
        }
        self._upsert(conn, "DocuFlowResult", job_id, row)

        chapter_t, keyword_t = self.tables["DocuFlowChapter"], self.tables["DocuFlowKeyword"]
        conn.execute(chapter_t.delete().where(chapter_t.c.JobId == job_id))
        conn.execute(keyword_t.delete().where(keyword_t.c.JobId == job_id))
        chapters = [
            {
                "JobId": job_id,
                "Ord": i,
                "ChapterNo": c.get("number"),
                "TitleVi": c.get("title_vi"),
                "TitleOriginal": c.get("title_original"),
                "Content": c.get("content"),
            }
            for i, c in enumerate(digest.get("chapters") or [], start=1)
        ]
        if chapters:
            conn.execute(chapter_t.insert(), chapters)
        kws = [
            {
                "JobId": job_id,
                "Ord": i,
                "Keyword": k.get("keyword"),
                "Display": k.get("display") or None,
                "Weight": k.get("weight"),
            }
            for i, k in enumerate(keywords, start=1)
        ]
        if kws:
            conn.execute(keyword_t.insert(), kws)


def apply_schema(engine, ddl: str) -> None:
    """Run a T-SQL script split on its ``GO`` batch separators."""
    batches = [b.strip() for b in re.split(r"^\s*GO\s*$", ddl, flags=re.M | re.I)]
    with engine.begin() as conn:
        for batch in batches:
            if batch:
                conn.exec_driver_sql(batch)
