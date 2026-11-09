#!/usr/bin/env python3
"""Copy the MinIO bucket to a local directory (used by scripts/backup.sh)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.object_storage import get_object_storage


def main(dest: str) -> None:
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    storage = get_object_storage()
    keys = storage.list_keys("")
    for key in keys:
        if not key or key.endswith("/"):
            continue
        target = dest_path / key
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as out:
            for chunk in storage.iter_stream(key, chunk_size=1024 * 1024):
                out.write(chunk)
    print(f"Copied {len(keys)} MinIO object(s) to {dest_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: backup_minio.py DEST_DIR")
    main(sys.argv[1])
