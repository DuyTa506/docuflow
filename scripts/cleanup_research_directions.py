#!/usr/bin/env python3
"""Tidy the shared `research_directions` catalog.

Two kinds of rubbish accumulate there:

* **Mồ côi** — a re-run deletes a document's associations and writes new ones,
  leaving the old direction rows attached to nothing.
* **Trùng lặp** — `direction_name` is UNIQUE and used to be matched by exact
  string, so "Giao thức PPP và HDLC" and "Giao thức HDLC và PPP" are two rows.

Predefined catalog entries are never touched. Dry-run by default; pass
``--apply`` to actually write.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.database import DatabaseManager
from data.db_models import DocumentResearchDirection, ResearchDirection
from utils.ctdt_catalog import name_key


def main():
    parser = argparse.ArgumentParser(description="Clean up the research-direction catalog")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = parser.parse_args()

    dbm = DatabaseManager()
    with dbm.session() as db:
        rows = db.query(ResearchDirection).all()
        assocs = db.query(DocumentResearchDirection).all()

        used_ids = {a.direction_id for a in assocs}
        by_doc = {(a.document_id, a.direction_id) for a in assocs}

        # ── Merge duplicates ─────────────────────────────────────────
        groups = defaultdict(list)
        for r in rows:
            groups[name_key(r.direction_name)].append(r)

        merged = 0
        moved: list[DocumentResearchDirection] = []
        for key, group in groups.items():
            if len(group) < 2:
                continue
            # Keep a predefined row if there is one, else the oldest.
            group.sort(key=lambda r: (not r.is_predefined, r.created_at or 0))
            keeper, *losers = group
            print(f"  ⋈ giữ «{keeper.direction_name}»")
            for loser in losers:
                print(f"      ← gộp «{loser.direction_name}»")
                # `ResearchDirection.document_research_directions` cascades
                # "all, delete-orphan". Re-pointing a row with
                # `a.direction_id = keeper.id` leaves it inside the loser's
                # relationship collection, so `db.delete(loser)` cascades and
                # deletes the row we just re-pointed — measured on the real
                # table: DOC_045 silently lost its direction. Delete the old
                # rows first, then insert fresh ones for the keeper.
                for a in [a for a in assocs if a.direction_id == loser.id]:
                    if args.apply:
                        db.delete(a)
                    if (a.document_id, keeper.id) in by_doc:
                        continue  # the document already points at the keeper
                    by_doc.add((a.document_id, keeper.id))
                    used_ids.add(keeper.id)
                    if args.apply:
                        moved.append(
                            DocumentResearchDirection(
                                document_id=a.document_id,
                                direction_id=keeper.id,
                                confidence=a.confidence,
                                reasoning=a.reasoning,
                            )
                        )
                if args.apply:
                    db.delete(loser)
                merged += 1

        # Added only after every delete has been flushed, so the cascade above
        # cannot reach them.
        if args.apply and moved:
            db.flush()
            for a in moved:
                db.add(a)

        # ── Drop orphans ─────────────────────────────────────────────
        orphans = [
            r
            for r in rows
            if not r.is_predefined
            and r.id not in used_ids
            and not any(r is g for grp in groups.values() for g in grp[1:] if len(grp) > 1)
        ]
        for r in orphans:
            if args.apply:
                db.delete(r)

        if not args.apply:
            db.rollback()

        print()
        print(f"tổng ban đầu       : {len(rows)}")
        print(f"predefined (giữ)   : {sum(1 for r in rows if r.is_predefined)}")
        print(f"gộp trùng lặp      : {merged}")
        print(f"xoá mồ côi         : {len(orphans)}")
        print(f"còn lại            : {len(rows) - merged - len(orphans)}")
        print()
        print("ĐÃ GHI." if args.apply else "DRY-RUN — chạy lại với --apply để thực sự ghi.")


if __name__ == "__main__":
    main()
