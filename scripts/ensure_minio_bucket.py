#!/usr/bin/env python3
"""Ensure the configured MinIO bucket exists (used by start.sh)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.object_storage import get_object_storage  # noqa: E402


def main() -> int:
    storage = get_object_storage()
    storage.ensure_bucket()
    print(f"MinIO bucket ready: {storage.bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
