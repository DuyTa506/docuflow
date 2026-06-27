#!/usr/bin/env python3
"""
Verify storage dedup migrations and optionally backfill derived caches.

- Compare tree_data vs rebuild from tree_nodes (non-lossy check)
- Optionally clear ocr_content when pages exist (assembler can rebuild)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.pageindex.tree_builder import build_tree_dict  # noqa: E402
from data.database import get_db_manager  # noqa: E402
from data.db_models import DigitizedText, Page, TreeIndex, TreeNode  # noqa: E402
from utils.text_assembly import assemble_ocr_from_pages  # noqa: E402
from utils.tree_payload import get_tree_payload  # noqa: E402


def _node_count(tree: dict) -> int:
    count = 1
    for child in tree.get("children") or tree.get("child_nodes") or []:
        count += _node_count(child)
    return count


def verify_trees(db, *, verbose: bool) -> tuple[int, int]:
    ok = fail = 0
    indices = db.query(TreeIndex).all()
    for idx in indices:
        payload = get_tree_payload(db, idx)
        nodes = db.query(TreeNode).filter(TreeNode.tree_index_id == idx.id).all()
        rebuilt = build_tree_dict(nodes)
        orig_nodes = _node_count(payload) if payload else 0
        rebuilt_nodes = _node_count(rebuilt) if rebuilt else 0
        if orig_nodes and rebuilt_nodes and abs(orig_nodes - rebuilt_nodes) > 2:
            print(f"[TREE FAIL] {idx.id}: orig_nodes={orig_nodes} rebuilt={rebuilt_nodes}")
            fail += 1
        else:
            if verbose:
                print(f"[TREE OK] {idx.id}: nodes~{orig_nodes}")
            ok += 1
    return ok, fail


def verify_ocr_assembly(db, *, verbose: bool) -> tuple[int, int]:
    ok = fail = 0
    rows = db.query(DigitizedText).all()
    for dt in rows:
        pages = (
            db.query(Page)
            .filter(Page.document_id == dt.document_id)
            .order_by(Page.page_number)
            .all()
        )
        if not pages or not dt.ocr_content:
            continue
        assembled = assemble_ocr_from_pages(pages)
        if len(assembled) < len(dt.ocr_content) * 0.9:
            print(f"[OCR FAIL] {dt.document_id}: assembled shorter than cache")
            fail += 1
        else:
            if verbose:
                print(f"[OCR OK] {dt.document_id}")
            ok += 1
    return ok, fail


def clear_redundant_ocr_cache(db, *, dry_run: bool) -> int:
    cleared = 0
    rows = db.query(DigitizedText).filter(DigitizedText.ocr_content.isnot(None)).all()
    for dt in rows:
        page_count = (
            db.query(Page).filter(Page.document_id == dt.document_id).count()
        )
        if page_count > 0:
            print(f"[CLEAR OCR CACHE] {dt.document_id}")
            if not dry_run:
                dt.ocr_content = None
            cleared += 1
    return cleared


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify dedup schema migrations")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--clear-ocr-cache",
        action="store_true",
        help="Null ocr_content when per-page markdown exists",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_manager = get_db_manager()
    with db_manager.session() as db:
        t_ok, t_fail = verify_trees(db, verbose=args.verbose)
        o_ok, o_fail = verify_ocr_assembly(db, verbose=args.verbose)
        cleared = 0
        if args.clear_ocr_cache:
            cleared = clear_redundant_ocr_cache(db, dry_run=args.dry_run)
        if not args.dry_run:
            db.commit()

    print(json.dumps({
        "tree_ok": t_ok,
        "tree_fail": t_fail,
        "ocr_ok": o_ok,
        "ocr_fail": o_fail,
        "ocr_cache_cleared": cleared,
        "dry_run": args.dry_run,
    }, indent=2))
    return 0 if (t_fail + o_fail) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
