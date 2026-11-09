"""The §2.2 unit label ("Chương N", "Phụ lục A") decided once, for the whole list.

Each entry used to be labelled on its own — its source ordinal when one was
parsed, else its list position — and the list position counted the "Các phần
bổ trợ" entry too. So a book whose first chapter follows the front matter
printed "Chương 2" for it, and unlabelled chapters between numbered ones
jumped backwards (DOC_013: 11 → 17 → 13).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.spatial.zone_classifier import split_chapter_heading

# The placeholder main_content writes for the grouped front/back matter entry.
AUX_TITLE_ORIGINAL = "Auxiliary sections"
AUX_KIND = "auxiliary"
# A unit with no number of its own that sits before chapter 1 (a preface).
UNNUMBERED_KIND = "unnumbered"

# "2. ", "1.2 ", "3) " at the start of a heading.
LEADING_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+")
# A single chapter number ("2. What is…"), not a section number ("2.1 …").
_CHAPTER_NUMBER_RE = re.compile(r"^\s*(\d+)[.)]?\s+(\S.*)$", re.S)


def split_numbered_heading(title: Optional[str]):
    """`split_chapter_heading`, also past Docling's stray number: ``6 Chapter 6: …``."""
    heading, name = split_chapter_heading(title)
    if heading is None and title and LEADING_NUMBER_RE.match(title):
        stripped_heading, stripped_name = split_chapter_heading(LEADING_NUMBER_RE.sub("", title))
        if stripped_heading is not None:
            return stripped_heading, stripped_name
    return heading, name


def _squash(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _adopt_leading_numbers(entries: List[Dict[str, Any]]) -> None:
    """ "2. What is…", "3. Architecture…" — a numbering scheme, so those are the
    chapter numbers. A lone "3. During testing…" among plain titles is not."""
    plain = [e for e in entries if not e.get("heading_kind")]
    numbered = [(e, _CHAPTER_NUMBER_RE.match(e["title_original"])) for e in plain]
    numbered = [(e, m) for e, m in numbered if m]
    ordinals = [int(m.group(1)) for _, m in numbered]
    if len(numbered) < 2 or 2 * len(numbered) < len(plain):
        return
    if any(b <= a for a, b in zip(ordinals, ordinals[1:])):
        return
    for entry, match in numbered:
        entry["heading_kind"] = "chapter"
        entry["heading_ordinal"] = int(match.group(1))
        entry["title_original"] = match.group(2).strip()


def normalize_chapter_entries(chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return copies with clean titles and one consistent numbering.

    - the auxiliary entry is marked as such and takes no number;
    - a label hidden behind a stray number is parsed and stripped;
    - a chapter without its own ordinal follows the previous chapter's;
    - `number` is the position among the real entries (BBKH numbering).
    """
    entries = [dict(c) for c in chapters]
    for entry in entries:
        entry["title_original"] = _squash(entry.get("title_original"))
        entry["title_vi"] = _squash(entry.get("title_vi"))
        suffix = f"({entry['title_original']})"
        if entry["title_original"] and entry["title_vi"].endswith(suffix):
            # The model wrote the bilingual form itself; the renderer adds it again.
            entry["title_vi"] = entry["title_vi"][: -len(suffix)].strip() or entry["title_vi"]
        if entry["title_original"] == AUX_TITLE_ORIGINAL:
            entry["heading_kind"] = AUX_KIND
            entry["heading_ordinal"] = None
        elif not entry.get("heading_kind"):
            heading, name = split_numbered_heading(entry["title_original"])
            if heading is not None and name:
                entry["heading_kind"], entry["heading_ordinal"] = heading[0], heading[2]
                entry["title_original"] = name

    _adopt_leading_numbers([e for e in entries if e.get("heading_kind") != AUX_KIND])

    stated = [
        (
            int(e["heading_ordinal"])
            if e.get("heading_kind") in (None, "chapter") and e.get("heading_ordinal")
            else None
        )
        for e in entries
    ]
    position = 0
    chapter_no = 0
    for i, entry in enumerate(entries):
        kind = entry.get("heading_kind")
        if kind == AUX_KIND:
            continue
        position += 1
        entry["number"] = position
        if kind not in (None, "chapter"):
            continue
        if stated[i]:
            chapter_no = stated[i]
            continue
        upcoming = next((n for n in stated[i + 1 :] if n), None)
        if upcoming is not None and chapter_no + 1 >= upcoming:
            # «Preface» before «Chapter 1»: any number it took would repeat one.
            entry["heading_kind"] = UNNUMBERED_KIND
            continue
        chapter_no += 1
        entry["heading_ordinal"] = chapter_no
    return entries
