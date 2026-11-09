#!/usr/bin/env python3
"""Pre-build download exports in MinIO (OCR, digest, summary, translation)."""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.database import get_db_manager
from data.db_models import Summary, Translation
from data.repositories import DocumentRepository, TranslationRepository
from services.export_service import export_service


async def warm(document_id: str, *, kinds: str = "all") -> None:
    dbm = get_db_manager()
    with dbm.session() as db:
        doc = DocumentRepository(db).get(document_id)
        if not doc:
            raise SystemExit(f"Document not found: {document_id}")
        print(f"Warming exports for {document_id} ({doc.title!r})…")

        if kinds in ("all", "ocr"):
            await export_service.cache_ocr_exports_after_extract(db, document_id)

        if kinds in ("all", "digest"):
            await export_service.cache_digest_export(document_id)

        if kinds in ("all", "summary"):
            summary = (
                db.query(Summary)
                .filter(
                    Summary.document_id == document_id,
                    Summary.status == "COMPLETED",
                )
                .order_by(Summary.created_at.desc())
                .first()
            )
            if summary and summary.content:
                await export_service.cache_summary_export(document_id, summary.id)
            else:
                print("  (skip summary — none completed)")

        if kinds in ("all", "translation"):
            trans = (
                db.query(Translation)
                .filter(
                    Translation.document_id == document_id,
                    Translation.status == "COMPLETED",
                )
                .order_by(Translation.created_at.desc())
                .first()
            )
            if trans:
                await export_service.cache_translation_exports(db, document_id, trans.id)
            else:
                print("  (skip translation — none completed)")

    print("Done. OCR/normalized DOCX+PDF and other kinds cached when applicable.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document_id", help="e.g. DOC_059")
    parser.add_argument(
        "--kinds",
        default="all",
        choices=["all", "ocr", "digest", "summary", "translation"],
        help="Which export families to warm",
    )
    args = parser.parse_args()
    asyncio.run(warm(args.document_id, kinds=args.kinds))


if __name__ == "__main__":
    main()
