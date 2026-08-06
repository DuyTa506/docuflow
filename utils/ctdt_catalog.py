"""CTĐT catalog — loading and matching for §3 of the digest.

Three things shape this module:

1. The catalog is the **national** one: Phụ lục I of Thông tư 09/2022/TT-BGDĐT,
   filtered down to the fields and groups within the Academy's scope
   (`scripts/build_ctdt_catalog.py`). It is a two-level tree — nhóm ngành
   (5-digit code) containing ngành (7-digit code) — split by ĐH/ThS/TS level.

2. **Codes identify, names do not.** In the national catalog 364 of 604
   discipline names sit under more than one code; the word "Khác" alone sits
   under 73. The previous version matched on the normalised name, which worked
   for the Academy's 42 programmes and is meaningless for this catalog. So
   `resolve_items` matches on the discipline code, and accepts a name only as a
   fallback when that name is unique within the level being resolved.

3. The catalog is **operational data, not code**. A deployment may carry its own
   catalog, or none at all. Loading therefore prefers an uploaded file, falls
   back to the bundled one in `config/`, and reports which it used — so callers
   can tell "no catalog loaded" apart from "no discipline fits".
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

CATALOG_KEYS = ("undergraduate", "master", "phd")

_BUNDLED_PATH = Path(__file__).resolve().parent.parent / "config" / "ctdt_catalog.json"

# Every dash character the catalog, the model and Vietnamese administrative
# prose have used for the same separator.
_DASHES = "‐‑‒–—―−-"
_DASH_RE = re.compile(rf"\s*[{_DASHES}]\s*")
_WS_RE = re.compile(r"\s+")
# Order matters: strip the longer prefix first.
_PREFIXES = ("chuyên ngành ", "nhóm ngành ", "ngành ", "chương trình ")

# A discipline code is exactly 7 digits. The model often returns "8520204 - Kỹ
# thuật rađa - dẫn đường" despite the prompt asking for codes only, so look for
# the code anywhere in the string.
_CODE_RE = re.compile(r"\b(\d{7})\b")


def empty_catalog() -> dict:
    return {k: [] for k in CATALOG_KEYS}


# ── Loading ──────────────────────────────────────────────────────────


def override_path() -> Path:
    """Where an uploaded catalog file lives.

    `CTDT_CATALOG_PATH` when set, otherwise
    `UPLOAD_DIR/catalog/ctdt_catalog.json`. Never the bundled file: an upload
    must not write into the repo.
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
    """Load the catalog; on any failure return an empty one rather than raising."""
    for path in (override_path(), _BUNDLED_PATH):
        data = _read_json(path)
        if data is not None:
            merged = empty_catalog()
            merged.update({k: v for k, v in data.items() if k in CATALOG_KEYS or k == "_source"})
            return merged
    return empty_catalog()


def _read_json(path: Path) -> dict | None:
    """Read *path* as a JSON object, or None (missing, corrupt or wrong type)."""
    try:
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("CTĐT catalog at %s is unreadable (%s) — ignoring", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("CTĐT catalog at %s is not a JSON object — ignoring", path)
        return None
    return data


# ── Tree traversal ───────────────────────────────────────────────────


def iter_programmes(catalog: dict, key: str) -> Iterator[tuple[str, str]]:
    """(code, name) for one level, in catalog order.

    Tolerates malformed data: `load_catalog` deliberately does not validate
    structure, so an upload with the wrong shape must leave §3 empty rather than
    bring the pipeline down.
    """
    for group in catalog.get(key) or []:
        if not isinstance(group, dict):
            continue
        for child in group.get("children") or []:
            if not isinstance(child, dict):
                continue
            code = str(child.get("code") or "").strip()
            name = str(child.get("name") or "").strip()
            if code and name:
                yield code, name


def count_programmes(catalog: dict, key: str) -> int:
    """Number of disciplines (leaves), not of groups."""
    return sum(1 for _ in iter_programmes(catalog, key))


def research_area_names(catalog: dict) -> list[str]:
    """Group names, de-duplicated across the three levels, in catalog order.

    Used as the "these are the areas the Academy works in" context for §3
    research directions. That slot used to hold 18 research-group names which the
    catalog file itself recorded as having no official name yet; the Thông tư 09
    groups are both real data and exactly what the Academy scope filter encodes.
    """
    names: list[str] = []
    seen: set[str] = set()
    for key in CATALOG_KEYS:
        for group in catalog.get(key) or []:
            if not isinstance(group, dict):
                continue
            if not group.get("children"):
                continue
            name = str(group.get("name") or "").strip()
            norm = normalize_name(name)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            names.append(name)
    return names


def has_entries(catalog: dict) -> bool:
    """True when at least one discipline exists at any level."""
    return any(count_programmes(catalog, k) > 0 for k in CATALOG_KEYS)


def catalog_text_block(catalog: dict, key: str) -> str:
    """One level's tree as prompt text: group, then its disciplines indented.

    Groups are present to orient the model — they say which area a discipline
    belongs to — but only a 7-digit code is a valid answer.
    """
    lines: list[str] = []
    for group in catalog.get(key) or []:
        if not isinstance(group, dict):
            continue
        children = [
            (str(c.get("code") or "").strip(), str(c.get("name") or "").strip())
            for c in group.get("children") or []
            if isinstance(c, dict)
        ]
        children = [(code, name) for code, name in children if code and name]
        if not children:
            continue
        lines.append(
            f"{str(group.get('code') or '').strip()} {str(group.get('name') or '').strip()}"
        )
        lines.extend(f"  {code} {name}" for code, name in children)
    return "\n".join(lines)


# ── Upload ───────────────────────────────────────────────────────────


def validate_catalog(data) -> dict:
    """Normalise an uploaded payload into the catalog structure.

    Unknown keys are dropped (except the free-text `_source` note). An entry
    missing a code or a name **raises** rather than being silently skipped —
    silent skipping is exactly how §3 ends up empty with nobody knowing why.
    Duplicate codes are merged, preserving order.
    """
    if not isinstance(data, dict):
        raise ValueError("Catalog must be a JSON object")

    cleaned = empty_catalog()
    for key in CATALOG_KEYS:
        if key not in data:
            continue
        groups = data[key]
        if not isinstance(groups, list):
            raise ValueError(f"Field '{key}' must be a list of groups")

        seen_groups: set[str] = set()
        out: list[dict] = []
        for group in groups:
            code, name = _entry_fields(group, key)
            if code in seen_groups:
                continue
            seen_groups.add(code)

            children_raw = group.get("children") or []
            if not isinstance(children_raw, list):
                raise ValueError(f"'children' of group {code} must be a list")

            seen_children: set[str] = set()
            children: list[dict] = []
            for child in children_raw:
                child_code, child_name = _entry_fields(child, key)
                if child_code in seen_children:
                    continue
                seen_children.add(child_code)
                children.append({"code": child_code, "name": child_name})

            out.append({"code": code, "name": name, "children": children})
        cleaned[key] = out

    source = data.get("_source")
    if isinstance(source, str) and source.strip():
        cleaned["_source"] = source.strip()
    return cleaned


def _entry_fields(entry, key: str) -> tuple[str, str]:
    """(code, name), whitespace-trimmed, or ValueError when either is missing."""
    if not isinstance(entry, dict):
        raise ValueError(f"Every entry in '{key}' must be an object with 'code' and 'name'")
    code = _WS_RE.sub(" ", str(entry.get("code") or "")).strip()
    name = _WS_RE.sub(" ", str(entry.get("name") or "")).strip()
    if not code or not name:
        raise ValueError(f"Entry in '{key}' is missing 'code' or 'name': {entry!r}")
    return code, name


def save_catalog(data) -> Path:
    """Validate then write the uploaded catalog. Returns the file written."""
    cleaned = validate_catalog(data)
    path = override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    logger.info("Saved CTĐT catalog to %s", path)
    return path


def clear_catalog() -> bool:
    """Delete the uploaded file so the bundled catalog takes effect again."""
    path = override_path()
    if not path.is_file():
        return False
    path.unlink()
    logger.info("Removed uploaded CTĐT catalog at %s", path)
    return True


# ── Matching ─────────────────────────────────────────────────────────


def normalize_name(name: str) -> str:
    """Comparison key: Unicode composition, dashes, whitespace, case, prefixes."""
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
    """Order-independent comparison key: normalised words, sorted.

    Research directions are free text, and "Giao thức PPP và HDLC" /
    "Giao thức HDLC và PPP" are the same direction. `normalize_name` cannot see
    that — measured against the real table it merged none of the 290 rows.

    Requiring the **same set of words** is deliberate: a longer, more specific
    direction keeps its own key. Merging two different directions is worse than
    leaving one duplicate behind.
    """
    return tuple(sorted(normalize_name(name).split()))


def _name_variants(name: str) -> tuple[str, str]:
    """Name lookup keys, strictest first: normalised, then whitespace removed."""
    norm = normalize_name(name)
    return norm, norm.replace(" ", "")


def resolve_items(catalog: dict, key: str, items: Iterable) -> tuple[list[str], list[str]]:
    """Match *items* against the disciplines of `catalog[key]`.

    Returns `(kept, dropped)`: `kept` holds the **official catalog names**, in
    catalog order, de-duplicated; `dropped` holds items that matched nothing, so
    the caller can log what the model invented.

    The 7-digit code wins. The name fallback applies only when that name is
    unique within the level being resolved: guessing at a code is worse than
    leaving the slot empty.
    """
    by_code: dict[str, str] = {}
    # Two name lookup tiers: exact normalised form, then whitespace removed. The
    # second tier makes "Kỹ thuật ra đa" match the official spelling "Kỹ thuật
    # rađa" — the same discipline, spaced differently.
    tiers: tuple[dict[str, str | None], ...] = ({}, {})
    for code, name in iter_programmes(catalog, key):
        by_code[code] = name
        for tier, variant in zip(tiers, _name_variants(name)):
            # None marks an ambiguous name — already seen under another code.
            tier[variant] = None if variant in tier else code

    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        if not isinstance(item, str) or not item.strip():
            dropped.append(item if isinstance(item, str) else repr(item))
            continue

        code = next((m for m in _CODE_RE.findall(item) if m in by_code), None)
        if code is None:
            variants = _name_variants(item)
            code = next(
                (c for tier, v in zip(tiers, variants) if (c := tier.get(v)) is not None), None
            )
        if code is None:
            dropped.append(item)
        elif code not in seen:
            seen.add(code)
            kept.append(by_code[code])
    return kept, dropped
