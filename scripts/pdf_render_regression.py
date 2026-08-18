"""Render a stored document's translation/OCR PDF and dump page PNGs + quality JSON.

Usage:
    python scripts/pdf_render_regression.py DOC_010 --kind translation --pages 1,2,3,6,11
    python scripts/pdf_render_regression.py DOC_010 --kind layout --pages 1,2,3,6,11
    python scripts/pdf_render_regression.py DOC_010 --kind ocr --pdf-mode facsimile
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_pages(raw: str) -> set[int]:
    return {int(p) for p in raw.split(",") if p.strip()}


def _raster(pdf_bytes: bytes, out: Path, wanted: set[int]) -> dict:
    import fitz

    doc_pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    source_index = 0
    texts = {}
    for i in range(doc_pdf.page_count):
        text = (doc_pdf[i].get_text() or "").replace("\xa0", " ")
        if text.lstrip().startswith("… (tiếp trang") or text.lstrip().startswith("... (tiếp trang"):
            continue
        source_index += 1
        if wanted and source_index not in wanted:
            continue
        pix = doc_pdf[i].get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        dest = out / f"page_{source_index:02d}.png"
        pix.save(str(dest))
        print("raster", dest)
        texts[source_index] = text
    manifest = {
        "pages": doc_pdf.page_count,
        "bytes": len(pdf_bytes),
        "search_preview": {str(k): v[:400] for k, v in texts.items()},
    }
    doc_pdf.close()
    return manifest


def _render_from_elements(db, doc, elements, *, pdf_mode: str, text_kind: str, lang: str):
    from core.pdf_render.renderer import render_document_pdf
    from data.repositories import DocumentRepository
    from services.export_service import export_service

    repo = DocumentRepository(db)
    pages = repo.get_pages(doc.id)
    orig_path, cleanup = export_service._resolve_original_pdf(doc)
    try:
        return render_document_pdf(
            pages=pages,
            elements=elements,
            original_pdf_path=orig_path,
            pdf_mode=pdf_mode,
            text_kind=text_kind,
            lang=lang,
        )
    finally:
        if cleanup and orig_path:
            import os

            if os.path.isfile(orig_path):
                os.remove(orig_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id")
    parser.add_argument(
        "--kind",
        choices=("translation", "ocr", "layout"),
        default="translation",
        help="translation=translated_elements when present; layout=stored OCR/Docling elements",
    )
    parser.add_argument("--pages", default="", help="comma-separated 1-based page numbers")
    parser.add_argument("--out", default="/tmp/pdf_render_regression")
    parser.add_argument("--pdf-mode", default="auto")
    args = parser.parse_args()

    from data.database import get_db_manager
    from data.db_models import Translation
    from data.repositories import DocumentRepository
    from services.export_service import export_service
    from utils.translation_elements import (
        deserialize_translated_elements,
        layout_element_to_dict,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dbm = get_db_manager()
    quality = None
    filename = f"{args.document_id}.{args.kind}.pdf"
    data = b""
    notes: list[str] = []

    with dbm.session() as db:
        repo = DocumentRepository(db)
        doc = repo.get(args.document_id)
        if not doc:
            print(f"document {args.document_id} not found", file=sys.stderr)
            return 1

        if args.kind == "ocr":
            data, filename, _ = export_service.build_ocr_export(
                db, doc, content_type="ocr", fmt="pdf", pdf_mode=args.pdf_mode
            )
        elif args.kind == "translation":
            trans = (
                db.query(Translation)
                .filter(
                    Translation.document_id == args.document_id, Translation.status == "COMPLETED"
                )
                .order_by(Translation.created_at.desc())
                .first()
            )
            if not trans:
                print("no completed translation", file=sys.stderr)
                return 1
            elems = []
            if trans.translated_elements:
                elems = deserialize_translated_elements(trans.translated_elements)
            notes.append(
                f"translation_id={trans.id} mode={trans.translation_mode} elements={len(elems)}"
            )
            if elems:
                mode = args.pdf_mode if args.pdf_mode != "auto" else "layout"
                result = _render_from_elements(
                    db,
                    doc,
                    elems,
                    pdf_mode=mode,
                    text_kind="translation",
                    lang=trans.target_language or "vi",
                )
                data = result.pdf_bytes
                quality = result.quality.to_dict()
                filename = f"{args.document_id}.{result.pdf_mode}.pdf"
                notes.append(f"quality.ok={result.quality.ok} fallback={result.quality.fallback}")
            else:
                notes.append(
                    "no translated_elements (likely pdf_overlay); falling back to layout elements"
                )
                layout_elems = repo.get_elements(doc.id)
                payloads = [
                    layout_element_to_dict(e, getattr(e.page, "page_number", 1) or 1)
                    for e in layout_elems
                ]
                mode = args.pdf_mode if args.pdf_mode != "auto" else "layout"
                result = _render_from_elements(
                    db,
                    doc,
                    payloads,
                    pdf_mode=mode,
                    text_kind="translation",
                    lang="en",
                )
                data = result.pdf_bytes
                quality = result.quality.to_dict()
                filename = f"{args.document_id}.layout-source.{result.pdf_mode}.pdf"
                notes.append(
                    f"layout_elements={len(payloads)} quality.ok={result.quality.ok} "
                    f"fallback={result.quality.fallback}"
                )
        else:
            layout_elems = repo.get_elements(doc.id)
            payloads = [
                layout_element_to_dict(e, getattr(e.page, "page_number", 1) or 1)
                for e in layout_elems
            ]
            mode = args.pdf_mode if args.pdf_mode != "auto" else "layout"
            result = _render_from_elements(
                db,
                doc,
                payloads,
                pdf_mode=mode,
                text_kind="translation",
                lang="en",
            )
            data = result.pdf_bytes
            quality = result.quality.to_dict()
            filename = f"{args.document_id}.layout.{result.pdf_mode}.pdf"
            notes.append(
                f"layout_elements={len(payloads)} quality.ok={result.quality.ok} "
                f"fallback={result.quality.fallback}"
            )

    pdf_path = out / filename
    pdf_path.write_bytes(data)
    print("wrote", pdf_path, "bytes", len(data))
    for note in notes:
        print(note)

    raster_info = _raster(data, out, _parse_pages(args.pages))
    manifest = {
        "document_id": args.document_id,
        "kind": args.kind,
        "filename": filename,
        "notes": notes,
        "quality": quality,
        **raster_info,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if quality:
        (out / "quality.json").write_text(
            json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        issues = quality.get("issues") or []
        critical = [i for i in issues if i.get("critical")]
        print(f"quality issues={len(issues)} critical={len(critical)}")
        for issue in critical[:12]:
            print(" critical", issue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
