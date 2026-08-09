"""The digest's "Thông tin quản trị CSDL" block.

Three free-text fields a librarian fills in — who wrote the digest, who
approved it, when it entered the catalogue. Nothing derives them, so they are
stored per document and rendered as-is.
"""

from __future__ import annotations

DIGEST_ADMIN_KEYS = ("reviewer", "reviewer_approved", "entry_date")


def digest_admin_defaults() -> dict:
    return {k: "" for k in DIGEST_ADMIN_KEYS}


def normalize_digest_admin(data) -> dict:
    """Validate a stored or uploaded admin block into its canonical shape."""
    if data is None:
        return digest_admin_defaults()
    if not isinstance(data, dict):
        raise ValueError("Thông tin quản trị phải là một JSON object")

    cleaned = digest_admin_defaults()
    for key in DIGEST_ADMIN_KEYS:
        if key not in data:
            continue
        value = data[key]
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Trường '{key}' phải là chuỗi")
        cleaned[key] = value.strip()
    return cleaned
