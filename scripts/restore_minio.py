#!/usr/bin/env python3
"""Upload a backup_minio.py directory back into the live bucket."""

from __future__ import annotations

import mimetypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.object_storage import get_object_storage


def main(src: str) -> None:
    src_path = Path(src)
    storage = get_object_storage()
    storage.ensure_bucket()
    count = 0
    for path in src_path.rglob("*"):
        if not path.is_file():
            continue
        key = str(path.relative_to(src_path)).replace("\\", "/")
        storage.put_file(
            key,
            str(path),
            content_type=mimetypes.guess_type(path.name)[0],
        )
        count += 1
    print(f"Restored {count} object(s) from {src_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: restore_minio.py SRC_DIR")
    main(sys.argv[1])
