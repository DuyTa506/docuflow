"""CTĐT catalog — loading and tolerant matching for digest §3.

Two facts shape this module:

1. The catalog is **operational data, not code**. A deployment may have the
   Academy's programme list, a partial list, or none at all. So loading
   resolves an uploaded override first, falls back to the file bundled in
   `config/`, and reports which one it used — callers decide what to do when
   there is nothing to match against.

2. Matching must be **tolerant of spelling, strict about membership**. The
   bundled entries carry a mandatory "Ngành " prefix and an en-dash (U+2013);
   a model answering "Kỹ thuật ra đa - dẫn đường" means the same programme.
   The old exact-string filter dropped it silently, which is how §3 came back
   empty for N4.11.160. Resolution therefore normalises both sides but always
   returns the **catalog's own spelling**, so nothing invented gets through.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

CATALOG_KEYS = ("undergraduate", "master", "phd", "strong_research_groups")

_BUNDLED_PATH = Path(__file__).resolve().parent.parent / "config" / "ctdt_catalog.json"

# Every Unicode dash the catalog, the models, and Vietnamese office documents
# have been observed to use for the same separator.
_DASHES = "‐‑‒–—―−-"
_DASH_RE = re.compile(rf"\s*[{_DASHES}]\s*")
_WS_RE = re.compile(r"\s+")
# Ordering matters: strip the longest prefix first.
_PREFIXES = ("chuyên ngành ", "nhóm ngành ", "ngành ", "chương trình ")


def empty_catalog() -> dict:
    return {k: [] for k in CATALOG_KEYS}


# ── Loading ──────────────────────────────────────────────────────────


def override_path() -> Path:
    """Where an uploaded catalog lives.

    `CTDT_CATALOG_PATH` if set, else `UPLOAD_DIR/catalog/ctdt_catalog.json`.
    Never the bundled file: an upload must not mutate the repo.
    """
    from config.settings import settings

    configured = (settings.ctdt_catalog_path or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(settings.upload_dir).expanduser() / "catalog" / "ctdt_catalog.json"


def catalog_source() -> str:
    """`"uploaded"` | `"bundled"` | `"none"` — which file load_catalog() will use."""
    if _read_json(override_path()) is not None:
        return "uploaded"
    if _read_json(_BUNDLED_PATH) is not None:
        return "bundled"
    return "none"


def load_catalog() -> dict:
    """Load the catalog, degrading to an empty one rather than raising."""
    for path in (override_path(), _BUNDLED_PATH):
        data = _read_json(path)
        if data is not None:
            merged = empty_catalog()
            merged.update({k: v for k, v in data.items() if k in CATALOG_KEYS or k == "_source"})
            return merged
    return empty_catalog()


def _read_json(path: Path) -> dict | None:
    """Parse *path* as a JSON object, or return None (missing/corrupt/wrong shape)."""
    try:
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("CTĐT catalog at %s is unreadable (%s) — ignoring it", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("CTĐT catalog at %s is not a JSON object — ignoring it", path)
        return None
    return data


def has_entries(catalog: dict) -> bool:
    """True when at least one programme/group is listed."""
    return any(catalog.get(k) for k in CATALOG_KEYS)


# ── Upload ───────────────────────────────────────────────────────────


def validate_catalog(data) -> dict:
    """Normalise an uploaded payload into the catalog shape.

    Unknown keys are dropped (except the free-text `_source` note); entries are
    trimmed, blank-filtered and de-duplicated while keeping their order.
    """
    if not isinstance(data, dict):
        raise ValueError("Catalog phải là một JSON object")

    cleaned = empty_catalog()
    for key in CATALOG_KEYS:
        if key not in data:
            continue
        items = data[key]
        if not isinstance(items, list):
            raise ValueError(f"Trường '{key}' phải là danh sách chuỗi")
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if not isinstance(item, str):
                continue
            name = _WS_RE.sub(" ", item).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        cleaned[key] = out

    source = data.get("_source")
    if isinstance(source, str) and source.strip():
        cleaned["_source"] = source.strip()
    return cleaned


def save_catalog(data) -> Path:
    """Validate and persist an uploaded catalog. Returns the file written."""
    cleaned = validate_catalog(data)
    path = override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    logger.info("CTĐT catalog saved to %s", path)
    return path


def clear_catalog() -> bool:
    """Remove the uploaded override so the bundled catalog applies again."""
    path = override_path()
    if not path.is_file():
        return False
    path.unlink()
    logger.info("CTĐT catalog override removed from %s", path)
    return True


# ── Matching ─────────────────────────────────────────────────────────


def normalize_name(name: str) -> str:
    """Comparison key: composition, dashes, spacing, case and prefix folded away."""
    text = unicodedata.normalize("NFC", str(name or ""))
    text = _DASH_RE.sub("-", text)
    text = _WS_RE.sub(" ", text).strip().strip(".,;:")
    text = text.casefold()
    for prefix in _PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text


def name_key(name: str | None) -> tuple[str, ...]:
    """Order-insensitive comparison key: the normalised words, sorted.

    Research directions arrive as free text, and "Giao thức PPP và HDLC" /
    "Giao thức HDLC và PPP" are the same direction. `normalize_name` cannot
    see that — measured on the live table it collapsed none of the 290 rows.

    Deliberately requires the *same set of words*: a longer, more specific
    direction keeps its own key. Merging two genuinely different directions
    would be worse than keeping a duplicate.
    """
    return tuple(sorted(normalize_name(name).split()))


def resolve_items(catalog: dict, key: str, items: Iterable) -> tuple[list[str], list[str]]:
    """Match *items* against `catalog[key]`.

    Returns `(kept, dropped)` where `kept` holds the **catalog's** spelling in
    catalog order-of-appearance, de-duplicated, and `dropped` holds the inputs
    that matched nothing — so a caller can log what the model made up.
    """
    lookup = {normalize_name(entry): entry for entry in catalog.get(key) or []}

    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        if not isinstance(item, str) or not item.strip():
            dropped.append(item if isinstance(item, str) else repr(item))
            continue
        match = lookup.get(normalize_name(item))
        if match is None:
            dropped.append(item)
        elif match not in seen:
            seen.add(match)
            kept.append(match)
    return kept, dropped
