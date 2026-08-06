"""Helpers for digest template formatting (bilingual keywords, chapter lines)."""

from __future__ import annotations

import re

# The digest template renders into Word paragraphs, which have no markdown.
# Anything the model emitted as markdown used to appear as literal characters.
_MD_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", flags=re.MULTILINE)
_MD_HEADING_TAIL_RE = re.compile(r"[ \t]+#{1,6}[ \t]*$", flags=re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", flags=re.DOTALL)
_MD_CODE_RE = re.compile(r"`([^`]+)`")
# Whitespace collapsing for the comparison keys below.
_WS = re.compile(r"\s+")


def plain_text(text: str | None) -> str:
    """Strip the markdown Word cannot render, leaving the words untouched.

    Deliberately conservative: headings, bold and inline code only. Single
    asterisks are left alone — in a technical digest they are far more likely
    to be multiplication than emphasis.
    """
    if not text:
        return ""
    out = _MD_HEADING_RE.sub("", str(text))
    out = _MD_HEADING_TAIL_RE.sub("", out)
    out = _MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2) or "", out)
    out = _MD_CODE_RE.sub(r"\1", out)
    return out.strip()


def strip_block_markdown(text: str | None) -> str:
    """Remove BLOCK markup only — `#` headings — and keep inline markup intact.

    `plain_text` throws away `**bold**` and `` `code` `` too, which was right
    while the digest could only emit flat strings. The renderer now builds Word
    runs, so inline markup should reach it and become real formatting; only the
    block-level markers, which have no run-level equivalent, are dropped here.
    """
    if not text:
        return ""
    out = _MD_HEADING_RE.sub("", str(text))
    out = _MD_HEADING_TAIL_RE.sub("", out)
    return out.strip()


def split_block_lines(text: str | None) -> list[str]:
    """Split into Word paragraphs, keeping inline markup for the renderer.

    Every line break becomes a paragraph boundary: a single `{{ abstract }}`
    placeholder collapsed the whole abstract into one unreadable block, because
    a newline inside a Word run renders as nothing at all.
    """
    return [line.strip() for line in strip_block_markdown(text).splitlines() if line.strip()]


def split_paragraphs(text: str | None) -> list[str]:
    """As above but flattened — for callers that can only take plain strings."""
    return [line.strip() for line in (plain_text(text)).splitlines() if line.strip()]


# Vietnamese label for each structural unit kind `match_chapter_heading` detects.
# Printing an appendix as "Chương 10" states a falsehood in an official document,
# and the unit kind is already known — nothing here has to be guessed.
_UNIT_LABELS = {"chapter": "Chương", "appendix": "Phụ lục", "part": "Phần", "section": "Mục"}


def _unit_label(kind: str | None, number: int, ordinal: int | None) -> str:
    label = _UNIT_LABELS.get(kind or "", "Chương")
    if kind == "appendix" and ordinal and 1 <= ordinal <= 26:
        # Appendices are lettered, and the source ordinal may be А/Б/В
        # (Cyrillic) — normalise to Latin so the Vietnamese edition reads.
        return f"{label} {chr(ord('A') + ordinal - 1)}"
    return f"{label} {ordinal or number}"


def chapter_heading(
    number: int,
    title_vi: str,
    title_original: str = "",
    *,
    doc_kind: str = "book",
    paper_count: int | None = None,
    heading_kind: str | None = None,
    heading_ordinal: int | None = None,
) -> str:
    """Opening line of a §2.2 entry, in the three forms the official template uses.

    - Book:                 ``Chương 1. Giới thiệu (Introduction).``
    - Proceedings, cluster: ``Khoa học máy tính (Computer Science), gồm 5 BBKH.``
    - Proceedings, single:  ``BBKH 1 - Giám sát chất lượng nước (Real-Time…).``

    ``paper_count`` < 2 falls back to the single form: "gồm 1 BBKH" is not a
    cluster.
    """
    vi = plain_text(title_vi)
    original = plain_text(title_original)
    # The extraction tree often returns the same string for both fields, and
    # printing it twice in parentheses is noise, not a bilingual pair.
    name = f"{vi} ({original})" if original and original != vi else vi

    if doc_kind == "proceedings":
        try:
            count = int(paper_count or 0)
        except (TypeError, ValueError):
            count = 0
        if count >= 2:
            return f"{name}, gồm {count} BBKH." if name else f"Cụm {number}, gồm {count} BBKH."
        return f"BBKH {number} - {name}." if name else f"BBKH {number}."

    label = _unit_label(heading_kind, number, heading_ordinal)
    return f"{label}. {name}." if name else f"{label}."


# A unit label at the very start of the body, with its parenthesis if present
# ("Phụ lục A (Приложение А) trình bày…"). The `^` anchor is deliberate: a label
# mid-sentence is a genuine cross-reference, not a restatement.
_RESTATED_LABEL_RE = re.compile(r"^\s*(Chương|Phụ lục|Phần|Mục)\s+(\S+?)\s*(\([^)]*\))?\s+(?=\S)")


def strip_restated_label(body: str | None, label: str | None) -> str:
    """Turn a restated label at the start of the body into a demonstrative.

    `chapter_heading` has already printed "Phụ lục A. Số nhị phân (Двоичные
    числа).", so a body opening with "Phụ lục A (Приложение А) trình bày…" says
    it twice — 8 of the 12 entries in N4.11.160 did.

    Only rewrites when the body restates **the unit being printed**: "Chương 8
    cũng bàn về…" inside chapter 2's body is a cross-reference and must stay.
    """
    text = (body or "").strip()
    if not text or not (label or "").strip():
        return text

    match = _RESTATED_LABEL_RE.match(text)
    if not match:
        return text

    kind, ordinal = match.group(1), match.group(2).rstrip(".,:")
    if f"{kind} {ordinal}".casefold() != label.strip().casefold():
        return text
    return f"{kind} này {text[match.end():]}"


# Self-reference markers. The set is closed on purpose: "chương" also opens
# "chương trình" (program), which appears on nearly every page of a computer
# architecture book, so requiring one of these followers makes that collision
# impossible rather than merely unlikely. An ordinal is deliberately absent —
# "Chương 5 đã trình bày" inside an appendix is a genuine cross-reference.
_SELF_REFERENCE_RE = re.compile(r"\b([Cc]hương)\s+(?=(?:này|cũng|còn)\b)")


def correct_unit_kind_words(body: str | None, kind: str | None) -> str:
    """Make an entry call its own unit by the right kind word.

    Both appendix entries in N4.11.160 that had a real summary described
    themselves as "Chương này" mid-body, having opened correctly with "Phụ lục
    này" — an official document calling appendix B a chapter.

    The kind is already known (the renderer prints "Phụ lục B" from it), so this
    corrects rather than guesses. Only self-references are touched.
    """
    text = body or ""
    label = _UNIT_LABELS.get(kind or "")
    if not text or not label or label == _UNIT_LABELS["chapter"]:
        return text

    def _replace(match: re.Match) -> str:
        word = label if match.group(1)[0].isupper() else label[0].lower() + label[1:]
        return f"{word} "

    return _SELF_REFERENCE_RE.sub(_replace, text)


# Both ASCII and full-width `（）` parentheses — Chinese/Japanese keyboards emit
# the latter, and §2.3 still uses the same "Vietnamese (original)" form.
_BILINGUAL_DISPLAY_RE = re.compile(r"^([^()（）]+?)\s*[(（]\s*([^()（）]+?)\s*[)）]$")


def collapse_bilingual_display(display: str | None) -> str:
    """`Core i7 (Core i7)` → `Core i7`.

    §2.3 displays "Vietnamese (original)". For proper nouns and part numbers —
    Core i7, OMAP4430, FPGA — both halves are equal and the repeated parenthesis
    is pure noise (5 of 20 entries in N4.11.160). Only the single-level `X (Y)`
    form is handled; anything with more parentheses is left alone rather than
    guessed at.
    """
    text = _WS.sub(" ", str(display or "")).strip()
    match = _BILINGUAL_DISPLAY_RE.match(text)
    if not match:
        return text
    left, right = match.group(1).strip(), match.group(2).strip()
    return left if left.casefold() == right.casefold() else text


# Han, kana, Hangul. The corpus is not only Russian.
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")
# Author-list separators: Latin comma/semicolon, their full-width forms, the CJK
# enumeration comma (、) and the katakana middle dot (・) between given and family
# name.
_AUTHOR_SPLIT_RE = re.compile(r"[(),;、，；・･]| и |\band\b")


def _min_ban_len(text: str) -> int:
    """Minimum length for a string to be distinctive enough to ban.

    Latin/Cyrillic needs 4, which rejects "and", "Tom" and initials. A complete
    CJK proper noun is only 2–3 characters — 王小明, 李华 — so a threshold of 4
    would miss every Chinese and Japanese author. Still 2 at the low end, so a
    single character is never banned.
    """
    return 2 if _CJK_RE.search(text) else 4


def drop_bibliographic_keywords(keywords: list[dict], bib: dict | None) -> list[dict]:
    """Drop keywords that are just the title or an author name — §1 has them.

    The keywords stage runs in parallel with biblio, so it cannot know the title
    or the authors; digest assembly is the only place that knows both. Matches
    against `keyword` as well as both halves of `display`, because the model puts
    the original-language form on either side.
    """
    if not keywords or not bib:
        return list(keywords or [])

    banned: set[str] = set()

    # Title: ban the whole string and each half of the bilingual form, but
    # **never** tokenise it. A book's subject overlapping its own title is
    # completely normal — banning per word would kill real subject terms.
    title = str(bib.get("title_display") or "").strip()
    if title:
        parts = [title, *re.split(r"[()（）]", title)]
        banned.update(p for p in (_norm_kw(x) for x in parts) if len(p) >= _min_ban_len(p))

    # Authors: ban the whole run and each proper noun in it, because a keyword
    # often carries only the surname ("Таненбаум" where §1 reads "Э. Таненбаум,
    # Т. Остин"). Initials fall below the length threshold.
    authors = str(bib.get("authors") or "").strip()
    if authors:
        chunks = _AUTHOR_SPLIT_RE.split(authors)
        banned.update(
            p for p in (_norm_kw(x) for x in [authors, *chunks]) if len(p) >= _min_ban_len(p)
        )
        for chunk in chunks:
            banned.update(
                p for p in (_norm_kw(w) for w in chunk.split()) if len(p) >= _min_ban_len(p)
            )

    kept = []
    for k in keywords:
        candidates = {_norm_kw(k.get("keyword"))}
        match = _BILINGUAL_DISPLAY_RE.match(_WS.sub(" ", str(k.get("display") or "")).strip())
        if match:
            candidates |= {_norm_kw(match.group(1)), _norm_kw(match.group(2))}
        else:
            candidates.add(_norm_kw(k.get("display")))
        if candidates & banned:
            continue
        kept.append(k)
    return kept


# Articles and classifiers that commonly open a Vietnamese noun phrase. Table of
# contents entries carry none, so "các kiến trúc máy tính song song" and the
# heading "Kiến trúc máy tính song song" must still reduce to one key.
_LEADING_ARTICLE_RE = re.compile(r"^(?:các|những|một số|sự|việc|về)\s+")


def _heading_key(text) -> str:
    return _LEADING_ARTICLE_RE.sub("", _norm_kw(text))


# Heading and keyword are translated by two separate LLM calls, so the same
# Russian phrase can arrive as "Mức kiến trúc tập lệnh" from one and "Cấp độ
# kiến trúc tập lệnh" from the other. Exact matching cannot see through that.
#
# Both numbers exist to protect genuine subjects, measured on N4.11.160:
# `số học nhị phân` keeps 3 of its 4 words inside `Số nhị phân` (0.75, under the
# bar) and `bộ nhớ ảo` shares nothing with any title, so both survive; a two-word
# keyword is too short for a ratio to carry meaning and still needs an exact hit.
_HEADING_OVERLAP_MIN_WORDS = 3
_HEADING_OVERLAP_RATIO = 0.8


def _repeats_heading(key: str, heading_words: list[frozenset[str]]) -> bool:
    words = frozenset(key.split())
    if len(words) < _HEADING_OVERLAP_MIN_WORDS:
        return False
    return any(
        len(words & heading) / len(words) >= _HEADING_OVERLAP_RATIO for heading in heading_words
    )


def drop_heading_keywords(keywords: list[dict], headings: list[str] | None) -> list[dict]:
    """Drop keywords that merely repeat a heading already printed in §2.2.

    §2.2 prints every heading verbatim together with a full summary paragraph,
    so repeating that exact string in §2.3 adds nothing — 6 of the 20 keywords in
    N4.11.160 did.

    Matches **exactly** after normalisation, never partially: "vi kiến trúc" is a
    real subject even though chapter 4 is titled "Mức vi kiến trúc", and "bộ nhớ
    ảo" is a real subject even though a whole chapter is about it.
    """
    if not keywords or not headings:
        return list(keywords or [])

    banned = {k for k in (_heading_key(h) for h in headings) if k}
    if not banned:
        return list(keywords)
    heading_words = [frozenset(k.split()) for k in banned]

    kept = []
    for k in keywords:
        candidates = {_heading_key(k.get("keyword"))}
        match = _BILINGUAL_DISPLAY_RE.match(_WS.sub(" ", str(k.get("display") or "")).strip())
        if match:
            candidates |= {_heading_key(match.group(1)), _heading_key(match.group(2))}
        else:
            candidates.add(_heading_key(k.get("display")))
        if candidates & banned:
            continue
        if any(c and _repeats_heading(c, heading_words) for c in candidates):
            continue
        kept.append(k)
    return kept


def _norm_kw(text) -> str:
    """Keyword comparison key: trim stray punctuation, collapse whitespace, casefold."""
    return _WS.sub(" ", str(text or "")).strip().strip(".,;:").casefold()


def join_catalog_items(items: list[str]) -> str:
    """Join CTĐT / NNC list for template field."""
    if not items:
        return ""
    return "; ".join(str(x).strip() for x in items if str(x).strip())


def is_chapter_schema(details: dict | None) -> bool:
    if not details or not isinstance(details, dict):
        return False
    chapters = details.get("chapters")
    return isinstance(chapters, list) and len(chapters) > 0


def bibliographic_defaults(title: str = "", pages: str | int | None = None) -> dict:
    return {
        "title_display": title or "",
        "authors": "",
        "publisher": "",
        "year": "",
        "isbn": "",
        "doi": "",
        "pages": str(pages) if pages is not None else "",
    }


def usage_scope_defaults() -> dict:
    return {
        "undergraduate": [],
        "master": [],
        "phd": [],
        "strong_research_groups": [],
    }
