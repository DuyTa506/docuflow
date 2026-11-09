"""Document domain profile and termbase — what the digest stages share.

The profile says which field / discipline / specialty a document belongs to, so
every stage writes in that specialty's Vietnamese register instead of translating
word by word. The termbase pins each recurring term to one Vietnamese rendering,
so «Interior Ballistics» is not «Nội đạn học» in §2.2 and «Đạn đạo học nội bộ» in §2.3.
"""

from __future__ import annotations

from typing import Any, Dict, List

LEVELS = ("field", "discipline", "specialty")
_LEVEL_LABELS = {"field": "lĩnh vực", "discipline": "ngành", "specialty": "chuyên ngành"}


def _bilingual(value: Any) -> Dict[str, str]:
    if isinstance(value, str) and value.strip():
        return {"vi": value.strip(), "en": value.strip()}
    if isinstance(value, dict):
        vi = str(value.get("vi") or "").strip()
        en = str(value.get("en") or "").strip()
        if vi or en:
            return {"vi": vi or en, "en": en or vi}
    return {}


def parse_profile(raw: Any) -> Dict[str, Any]:
    """Normalise the model's answer; anything unusable gives ``{}``."""
    if not isinstance(raw, dict):
        return {}
    profile: Dict[str, Any] = {}
    for level in LEVELS:
        value = _bilingual(raw.get(level))
        if value:
            profile[level] = value
    clusters = [c for c in map(_bilingual, raw.get("clusters") or []) if c]
    if clusters:
        profile["clusters"] = clusters
    summary = str(raw.get("summary_en") or "").strip()
    if profile and summary:
        profile["summary_en"] = summary
    return profile


def domain_clause(profile: Any) -> str:
    """Prompt lines naming the document's domain (or "" when it is unknown)."""
    if not isinstance(profile, dict):
        return ""
    parts = [
        f"{_LEVEL_LABELS[level]} {profile[level]['vi']} ({profile[level]['en']})"
        for level in LEVELS
        if profile.get(level)
    ]
    if not parts:
        return ""
    clusters = [c["vi"] for c in profile.get("clusters") or []]
    spread = f" The volume spans several areas: {'; '.join(clusters)}." if clusters else ""
    return (
        f"DOCUMENT DOMAIN: {'; '.join(parts)}.{spread}\n"
        "Use the standard Vietnamese terminology of this specialty, as a specialist in "
        "the field would write it. Do NOT translate word by word.\n"
    )


def parse_termbase(rows: Any) -> Dict[str, Dict[str, str]]:
    """``[{"src","vi","en"}]`` → ``{src: {"vi","en"}}``; rows that translate nothing are dropped."""
    termbase: Dict[str, Dict[str, str]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        src = str(row.get("src") or "").strip()
        vi = str(row.get("vi") or "").strip()
        if not src or not vi or src.casefold() == vi.casefold():
            continue
        termbase[src] = {"vi": vi, "en": str(row.get("en") or "").strip() or src}
    return termbase


def merge_termbase(
    existing: Any, generated: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, str]]:
    """The term table after a rerun: entries already there win, new terms are added.

    A correction made through the API must outlive the next pipeline run, and a fresh
    model answer is no more trustworthy than the one it would replace.
    """
    kept = existing if isinstance(existing, dict) else {}
    return {**generated, **kept}


def termbase_glossary(termbase: Any) -> Dict[str, str]:
    """The ``{source term: Vietnamese}`` view that ``utils.glossary`` prompts use."""
    if not isinstance(termbase, dict):
        return {}
    return {src: v["vi"] for src, v in termbase.items() if isinstance(v, dict) and v.get("vi")}


def translation_glossary(keyword_displays: Any, termbase: Any) -> Dict[str, str]:
    """Glossary for translating the document: the keywords' renderings, then the termbase
    over them — it is the one place a term's Vietnamese is decided."""
    from utils.glossary import parse_glossary

    return {**parse_glossary(keyword_displays), **termbase_glossary(termbase)}
