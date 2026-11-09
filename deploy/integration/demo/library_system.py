#!/usr/bin/env python3
"""A stand-in for the library's management system, for the integration demo.

It plays the library's side of the contract and uses nothing from DocuFlow's
code — only HTTP and its own SQL Server database — so it doubles as reference
client code for the library's developers:

  setup   create LibraryDB, the library's catalogue table dbo.TaiLieu (seeded
          with the demo documents) and DocuFlow's result tables (mssql_schema.sql)
  submit  send every dbo.TaiLieu row still CHO_GUI to POST /jobs
  listen  receive signed callbacks, verify them, pull the result, update dbo.TaiLieu
  show    print the catalogue joined with what DocuFlow wrote into docuflow.*
  run     setup + submit + listen until every job is final + show
  reset   drop LibraryDB, to run the demo again from the start

Settings come from the environment (start_demo.sh sets them):
  DOCUFLOW_URL, DOCUFLOW_API_KEY, DOCUFLOW_WEBHOOK_SECRET,
  LIBRARY_MSSQL_URL (…/master; LibraryDB is created next to it), CALLBACK_PORT
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import sqlalchemy as sa

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCHEMA_SQL = HERE.parent / "mssql_schema.sql"

DOCUFLOW_URL = os.environ.get("DOCUFLOW_URL", "http://localhost:8023")
API_KEY = os.environ.get("DOCUFLOW_API_KEY", "")
WEBHOOK_SECRET = os.environ.get("DOCUFLOW_WEBHOOK_SECRET", "")
MASTER_URL = os.environ.get(
    "LIBRARY_MSSQL_URL", "mssql+pymssql://sa:DocuFlow#Demo2026@localhost:14330/master"
)
LIBRARY_URL = MASTER_URL.rsplit("/", 1)[0] + "/LibraryDB"
CALLBACK_PORT = int(os.environ.get("CALLBACK_PORT", "9099"))
CALLBACK_URL = os.environ.get("CALLBACK_URL", f"http://localhost:{CALLBACK_PORT}/docuflow/callback")

# The library's own catalogue: which file, which language, which services it wants.
SEED = [
    ("TL-2026-0001", "Novelty in the military domain", "en",
     "datasets/extraction/sources/en/en_quan_su_novelty_military_domain.pdf", "ocr,digest,translate"),
    ("TL-2026-0002", "Giáo dục song ngữ cho học sinh Khmer", "vi",
     "datasets/extraction/sources/vi/vi_giao_duc_khmer_bilingual_education.pdf", "ocr,digest,translate"),
]

LIBRARY_DDL = """
IF OBJECT_ID(N'dbo.TaiLieu', N'U') IS NULL
CREATE TABLE dbo.TaiLieu (
    MaTaiLieu     NVARCHAR(20)  NOT NULL PRIMARY KEY,
    NhanDe        NVARCHAR(500) NOT NULL,
    NgonNgu       NVARCHAR(10)  NOT NULL,
    DuongDanTep   NVARCHAR(500) NOT NULL,
    YeuCau        NVARCHAR(100) NOT NULL,
    TrangThaiXuLy NVARCHAR(30)  NOT NULL DEFAULT N'CHO_GUI',
    DocuFlowJobId NVARCHAR(32)  NULL,
    CapNhatLuc    DATETIME2     NULL
);
GO
CREATE OR ALTER VIEW dbo.vw_TaiLieu_DocuFlow AS
SELECT t.MaTaiLieu, t.NhanDe, t.TrangThaiXuLy,
       j.JobId, j.Status, j.OcrStatus, j.DigestStatus, j.TranslateStatus,
       r.Title, r.Authors, r.PublishYear, r.Keywords,
       LEN(r.OcrText) AS OcrChars, LEN(r.TranslationText) AS TranslationChars,
       (SELECT COUNT(*) FROM docuflow.DocuFlowChapter c WHERE c.JobId = j.JobId) AS Chapters
FROM dbo.TaiLieu t
LEFT JOIN docuflow.DocuFlowJob j ON j.JobId = t.DocuFlowJobId
LEFT JOIN docuflow.DocuFlowResult r ON r.JobId = t.DocuFlowJobId;
"""


def log(msg: str) -> None:
    print(f"[thư viện {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_batches(engine, sql: str) -> None:
    with engine.begin() as conn:
        for batch in re.split(r"^\s*GO\s*$", sql, flags=re.M | re.I):
            if batch.strip():
                conn.exec_driver_sql(batch)


def library_db():
    return sa.create_engine(LIBRARY_URL, pool_pre_ping=True)


# ── setup ─────────────────────────────────────────────────────────────


def setup() -> None:
    master = sa.create_engine(MASTER_URL, isolation_level="AUTOCOMMIT")
    with master.connect() as conn:
        conn.exec_driver_sql("IF DB_ID(N'LibraryDB') IS NULL CREATE DATABASE LibraryDB")
    engine = library_db()
    run_batches(engine, SCHEMA_SQL.read_text(encoding="utf-8"))
    run_batches(engine, LIBRARY_DDL)
    with engine.begin() as conn:
        for row in SEED:
            conn.execute(
                sa.text(
                    "IF NOT EXISTS (SELECT 1 FROM dbo.TaiLieu WHERE MaTaiLieu = :id) "
                    "INSERT INTO dbo.TaiLieu (MaTaiLieu, NhanDe, NgonNgu, DuongDanTep, YeuCau) "
                    "VALUES (:id, :title, :lang, :path, :tasks)"
                ),
                dict(zip(("id", "title", "lang", "path", "tasks"), row)),
            )
    log("LibraryDB sẵn sàng: dbo.TaiLieu + schema docuflow")


def reset() -> None:
    master = sa.create_engine(MASTER_URL, isolation_level="AUTOCOMMIT")
    with master.connect() as conn:
        conn.exec_driver_sql(
            "IF DB_ID(N'LibraryDB') IS NOT NULL BEGIN "
            "ALTER DATABASE LibraryDB SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE LibraryDB; END"
        )
    log("đã xoá LibraryDB")


# ── submit ────────────────────────────────────────────────────────────


def submit() -> list[str]:
    engine = library_db()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text("SELECT MaTaiLieu, NhanDe, NgonNgu, DuongDanTep, YeuCau FROM dbo.TaiLieu "
                    "WHERE TrangThaiXuLy = N'CHO_GUI'")
        ).all()
    jobs = []
    for ma, title, lang, path, tasks in rows:
        with open(ROOT / path, "rb") as fh:
            resp = httpx.post(
                f"{DOCUFLOW_URL}/api/v2/integration/jobs",
                headers={"X-API-Key": API_KEY},
                data={"external_id": ma, "title": title, "source_language": lang,
                      "tasks": tasks, "callback_url": CALLBACK_URL},
                files={"file": (Path(path).name, fh, "application/pdf")},
                timeout=120,
            )
        if resp.status_code != 202:
            log(f"{ma}: gửi thất bại HTTP {resp.status_code} {resp.text[:200]}")
            continue
        job = resp.json()
        jobs.append(job["job_id"])
        with engine.begin() as conn:
            conn.execute(
                sa.text("UPDATE dbo.TaiLieu SET TrangThaiXuLy = N'DA_GUI', DocuFlowJobId = :job, "
                        "CapNhatLuc = SYSUTCDATETIME() WHERE MaTaiLieu = :ma"),
                {"job": job["job_id"], "ma": ma},
            )
        log(f"{ma} → {job['job_id']} ({', '.join(job['tasks'])}), trạng thái {job['status']}")
    return jobs


# ── callback receiver ─────────────────────────────────────────────────


def verify(secret: str, timestamp: str, body: bytes, signature: str, tolerance: int = 300) -> bool:
    """HMAC-SHA256 over "<timestamp>." + body, compared in constant time."""
    try:
        if abs(time.time() - int(timestamp)) > tolerance:
            return False
    except (TypeError, ValueError):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), f"{timestamp}.".encode() + body,
                                    hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


FINAL_STATES = {"COMPLETED": "HOAN_THANH", "PARTIAL": "HOAN_THANH_MOT_PHAN", "FAILED": "LOI"}
done: dict[str, str] = {}


class CallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not verify(WEBHOOK_SECRET, self.headers.get("X-DocuFlow-Timestamp"), body,
                      self.headers.get("X-DocuFlow-Signature")):
            log("callback bị từ chối: chữ ký không hợp lệ")
            self.send_response(401)
            self.end_headers()
            return
        event = json.loads(body)
        job = event["job"]
        log(f"callback {event['event']} cho {job['external_id']} ({job['job_id']}): {job['stages']}")

        # The callback is only a notice; the data is pulled from the API.
        result = httpx.get(job["links"]["result"], headers={"X-API-Key": API_KEY}, timeout=60).json()
        digest = result.get("digest") or {}
        log(f"  kết quả: OCR {len((result.get('ocr') or {}).get('text') or '')} ký tự, "
            f"{len(digest.get('chapters') or [])} chương, {len(digest.get('keywords') or [])} từ khoá, "
            f"bản dịch {len((result.get('translation') or {}).get('text') or '')} ký tự")

        with library_db().begin() as conn:
            conn.execute(
                sa.text("UPDATE dbo.TaiLieu SET TrangThaiXuLy = :st, CapNhatLuc = SYSUTCDATETIME() "
                        "WHERE DocuFlowJobId = :job"),
                {"st": FINAL_STATES.get(job["status"], job["status"]), "job": job["job_id"]},
            )
        done[job["job_id"]] = job["status"]
        self.send_response(204)
        self.end_headers()

    def log_message(self, *args):
        pass


def listen(expected: list[str] | None = None, timeout_s: int = 4 * 3600) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", CALLBACK_PORT), CallbackHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"đang chờ callback tại {CALLBACK_URL}")
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            if expected is not None and all(j in done for j in expected):
                break
            time.sleep(2)
    finally:
        server.shutdown()


# ── show ──────────────────────────────────────────────────────────────


def show() -> None:
    with library_db().connect() as conn:
        rows = conn.execute(sa.text("SELECT * FROM dbo.vw_TaiLieu_DocuFlow ORDER BY MaTaiLieu")).mappings().all()
        for r in rows:
            print(json.dumps(dict(r), ensure_ascii=False, default=str, indent=1))
        counts = {
            t: conn.execute(sa.text(f"SELECT COUNT(*) FROM docuflow.{t}")).scalar()
            for t in ("DocuFlowJob", "DocuFlowResult", "DocuFlowChapter", "DocuFlowKeyword")
        }
    print("docuflow.* rows:", counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("setup", "submit", "listen", "show", "run", "reset"))
    cmd = parser.parse_args().cmd
    if cmd == "setup":
        setup()
    elif cmd == "submit":
        submit()
    elif cmd == "listen":
        listen()
    elif cmd == "show":
        show()
    elif cmd == "reset":
        reset()
    else:
        setup()
        jobs = submit()
        if jobs:
            listen(jobs)
        show()


if __name__ == "__main__":
    main()
