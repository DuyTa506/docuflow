"""Pick digest §2.2 units out of a spatial tree whose levels are unreliable.

The tree's *hierarchy* is noise: `calculate_adaptive_thresholds` assigns levels
by percentile, so ~5% of all elements become level 0 no matter what the document
actually contains — an 816-page book yielded 265 "chapters". Its *reading order*,
however, is exact, because `build_tree_from_elements` attaches elements in the
order they were read.

So a unit is a contiguous reading-order span between two structural anchors.
That is immune to every level defect at once, including the common case where a
chapter's subsections ended up as its siblings rather than its children.

Anchors are found by cascading tiers, textual evidence first:

    max(chapter_vocabulary, section_prefix) → section_vocabulary
        → numbered_sections → root_children → mass_segmentation

The last tier always succeeds, so there is no "no structure found" failure mode —
but the tier that fired is reported in the metadata and surfaced as a quality
warning, so a machine-cut digest is never silently passed off as a real outline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import sqrt
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from config.settings import settings
from core.spatial.zone_classifier import (
    match_chapter_heading,
    parse_section_ordinal,
    split_chapter_heading,
)

# Labels that describe content, not structure. Blacklist rather than whitelist:
# on OCR'd PDFs virtually every element is labelled "text", so a heading
# whitelist would select nothing. "paragraph" is what thinning assigns to merged
# body blocks.
NON_ANCHOR_LABELS = frozenset(
    {
        "paragraph",
        "table",
        "figure",
        "image",
        "chart",
        "graph",
        "picture",
        "equation",
        "formula",
        "isolate_formula",
        "caption",
        "footnote",
        "page_header",
        "page_footer",
        "header",
        "footer",
    }
)

# Above this a "title" is a paragraph that thinning promoted, not a heading.
ANCHOR_TITLE_MAX_CHARS = 150
# A node whose title *is* its body, beyond this length, is a thinning artifact.
TITLE_IS_BODY_MIN_CHARS = 80
# Table-of-contents / index detection: a run of short lines ending in a page
# number. Language-independent, so it catches Оглавление, Mục lục and
# back-of-book indexes alike.
TOC_LINE_MAX_CHARS = 120
TOC_RUN_MIN = 8
# Front matter only becomes its own unit when it carries real weight.
PROLOGUE_MIN_RATIO = 0.02
# Granularity a reader perceives as "a chapter" when a document has no headings.
TARGET_UNIT_CHARS = 20_000

# Adaptive §2.2 ceiling. Anchored at "a ~400-page book earns at most 25 entries"
# and scaled by the square root of length: a document twice as long warrants more
# entries, but not twice as many — a digest stays a digest.
#
# Division of labour: on short documents the absolute MIN_UNIT_CHARS floor is what
# prevents fragmentation, so the ceiling must not bite there — a 20-page report
# with 8 real sections is not a defect. The ceiling exists for long documents,
# where units can each clear the char floor and still number in the hundreds
# (N4.11.160: 265 units of ~7k chars each).
CAP_REFERENCE_CHARS = 1_000_000
CAP_UNITS_AT_REFERENCE = 25
CAP_MIN_UNITS = 10
CAP_MAX_UNITS = 40

# Strong structural kinds. "section" is deliberately excluded: a book with both
# Глава and Раздел must anchor on the chapters, not fragment on the sections.
STRONG_KINDS = frozenset({"chapter", "part", "appendix"})

_TOC_TAIL_RE = re.compile(r"(?:\.{2,}|\s)\d{1,4}\s*$")

# Titles that can never open a §2.2 unit. Each one became a "chapter" in the
# 2026-09 E2E run: formulas, list items, chapter-end footnotes, table/figure
# references and sentence lead-ins promoted to titles by tree thinning.
_IMPLAUSIBLE_HEADING_RES = (
    re.compile(r"\\[(\[]|\$|\^\{|_\{"),  # LaTeX
    re.compile(r"^\s*(?:\d{1,3}|[a-zа-яё])\)\s"),  # "4) Стравливание …", "a) F1"
    re.compile(r"^\s*\d+\s+\d+\s+\S"),  # footnote "1 7 These and other …"
    re.compile(r"\b(?:табл(?:ица)?|рис(?:унок)?|fig(?:ure)?|table|bảng|hình)\.?\s*\d", re.I),
    re.compile(r"[,:;]\s*$"),  # "Indeed:", "…is given by,"
    re.compile(r"\s=\s"),
)
# Lines that sit before a chapter's first section but never name the chapter:
# an exercise, a quoted question, a sentence cut off mid-clause (E2E 2026-09:
# «Problem 2», «3. During testing: “Can I tell…?”», «The evaluation of which yields»).
_NOT_A_CHAPTER_NAME_RES = (
    re.compile(
        r"^\s*(?:problem|exercise|example|solution|задача|пример|упражнение|bài\s+tập|ví\s+dụ)"
        r"\s*\d",
        re.I,
    ),
    re.compile(r"[“\"«][^”\"»]*\?"),
    re.compile(
        r"\s(?:which|that|of|and|or|the|a|an|to|by|with|as|for|in|on|from|at|be|is|are|"
        r"yields|gives|becomes)$",
        re.I,
    ),
)
# Headings a chapter closes with. Their recurrence, one cluster per chapter,
# outlines the chapters when no chapter heading survived OCR (CompTIA PenTest+:
# 12 × Summary / Exam Essentials / Lab Exercises / Review Questions).
_END_MARKER_HEADINGS = frozenset(
    {
        "summary",
        "chapter summary",
        "exam essentials",
        "review questions",
        "lab exercises",
        "key terms",
        "where to go from here?",
        "where to go from here",
        "контрольные вопросы",
        "вопросы для самопроверки",
        "задачи и вопросы",
        "câu hỏi ôn tập",
        "本章小结",
        "习题",
        "思考题",
    }
)
END_MARKER_MIN_CHAPTERS = 3
# Closings name chapters worse than surviving chapter headings (they borrow the
# next heading), so they only win when they find clearly more chapters.
END_MARKER_MIN_ADVANTAGE = 1.5
# After a closing: a review question or a numbered answer is not a chapter opening.
_NOT_AN_OPENING_RE = re.compile(r"\?\s*$|^\s*\d+\.\s+\S")
# Closing sections of one chapter sit within this much text of each other.
END_MARKER_CLUSTER_GAP = 12000
_CONTENTS_HEADINGS = frozenset(
    {"contents", "table of contents", "содержание", "оглавление", "mục lục", "目录", "目 录"}
)
# A contents line's trailing page number and dot leaders: «Глава 2. Полупроводники. 25».
_TOC_PAGE_TAIL_RE = re.compile(r"(?:[\s.·…_]*\d{1,4})?\s*$")
_TOC_PLAIN_CHAPTER_RE = re.compile(r"^\s*(\d{1,2})\.?\s+([^\d\s].{2,})$")
# Navigation and end-of-chapter sections: real headings, never a unit of their own.
_NON_UNIT_HEADINGS = frozenset(
    {
        "contents",
        "table of contents",
        "содержание",
        "оглавление",
        "mục lục",
        "目录",
        "目 录",
        "review questions",
        "exercises",
        "problems",
        "references",
        "контрольные вопросы",
        "задачи и вопросы",
        "вопросы для самопроверки",
        "список литературы",
        "câu hỏi ôn tập",
        "câu hỏi kiểm tra",
        "tài liệu tham khảo",
        "this page intentionally left blank",
    }
)
# Front matter directly before «1.1» is not chapter 1's name (Ru_Book).
_FRONT_MATTER_TITLES = frozenset(
    {
        "preface",
        "foreword",
        "acknowledgements",
        "acknowledgments",
        "предисловие",
        "от автора",
        "lời nói đầu",
        "lời giới thiệu",
        "前言",
        "序",
    }
)
# "2 Chapter 2: Overview" — a running number glued in front of the heading.
_REDUNDANT_NUMBER_RE = re.compile(r"^(\d{1,3})\s+(?=\D)")
# Structural headings closer together than this are a table of contents.
TOC_HEADING_RUN_MIN = 3
# Share of section-prefix runs that must climb monotonically to count as chapters.
PREFIX_MONOTONIC_MIN = 0.8

_CONTENT_KEYS = ("content", "text", "text_content", "text_full")
_CHILD_KEYS = ("children", "child_nodes", "nodes")


def effective_max_units(total_chars: int) -> int:
    """How many §2.2 entries this document may have.

    `MAIN_CONTENT_MAX_UNITS` > 0 pins the ceiling (use it for a work that
    genuinely has more chapters than the curve allows); 0 — the default — scales
    it with document length.
    """
    override = settings.main_content_max_units
    if override and override > 0:
        return int(override)
    scaled = CAP_UNITS_AT_REFERENCE * sqrt(max(0, total_chars) / CAP_REFERENCE_CHARS)
    return max(CAP_MIN_UNITS, min(CAP_MAX_UNITS, round(scaled)))


def _own_text(node: Dict[str, Any]) -> str:
    for key in _CONTENT_KEYS:
        value = node.get(key)
        if value:
            return str(value)
    return ""


def _child_nodes(node: Dict[str, Any]) -> List[dict]:
    for key in _CHILD_KEYS:
        value = node.get(key)
        if value:
            return list(value)
    return []


@dataclass
class _Ref:
    """One tree node, flattened into reading order."""

    node: Dict[str, Any]
    index: int
    title: str
    text: str
    label: str
    page: Optional[int]
    is_root_child: bool
    is_toc: bool = False
    anchor_eligible: bool = False

    @property
    def mass(self) -> int:
        return len(self.text)


@dataclass
class _Unit:
    title: str
    refs: List[_Ref] = field(default_factory=list)
    # The anchor's page, captured at build time: thin front matter gets folded
    # into unit 1, and that must not drag the unit's page back to the cover.
    anchor_page: Optional[int] = None
    # The chapter number its sections state (3.1, 3.2 … → 3), when known.
    chapter_ordinal: Optional[int] = None

    @property
    def mass(self) -> int:
        return sum(r.mass for r in self.refs)

    @property
    def page(self) -> int:
        if self.anchor_page:
            return self.anchor_page
        for ref in self.refs:
            if ref.page:
                return ref.page
        return 1

    def to_node(self) -> Dict[str, Any]:
        """Materialise as a tree node so existing helpers work unchanged.

        Children are shallow and flat — each node's *own* text appears exactly
        once, in reading order — so gather_node_text/build_stratified_sample do
        not double-count a parent and its descendants.
        """
        return {
            "title": self.title,
            "content": "",
            "page_number": self.page,
            **({"chapter_ordinal": self.chapter_ordinal} if self.chapter_ordinal else {}),
            # Every heading inside the unit, complete and in reading order. On a
            # kỷ yếu these are the candidate BBKH titles — a heuristic cannot
            # tell a paper title from a section heading inside a paper, but the
            # list is short enough to hand to the model whole rather than
            # sampled, which is what makes counting them answerable at all.
            "member_titles": [
                ref.title for ref in self.refs[1:] if ref.anchor_eligible and ref.title
            ],
            "children": [
                {
                    "title": ref.title,
                    "content": ref.text,
                    "page_number": ref.page or 1,
                    "children": [],
                }
                for ref in self.refs
            ],
        }


def _flatten_preorder(tree: Any) -> List[_Ref]:
    """Pre-order DFS == reading order, regardless of how wrong the levels are."""
    if isinstance(tree, list):
        roots = list(tree)
    elif isinstance(tree, dict):
        roots = _child_nodes(tree)
        # A childless wrapper is only a unit when it carries text of its own —
        # a bare {"title": ..., "children": []} has nothing to summarize.
        if not roots and _own_text(tree):
            roots = [tree]
    else:
        return []

    refs: List[_Ref] = []

    def walk(node: Dict[str, Any], is_root_child: bool) -> None:
        if not isinstance(node, dict):
            return
        page = node.get("page_number")
        refs.append(
            _Ref(
                node=node,
                index=len(refs),
                title=repair_cp1251_mojibake(str(node.get("title") or "").strip()),
                text=_own_text(node).strip(),
                label=str(node.get("label") or "").strip().lower(),
                page=int(page) if isinstance(page, (int, float)) and page else None,
                is_root_child=is_root_child,
            )
        )
        for child in _child_nodes(node):
            walk(child, False)

    for root in roots:
        walk(root, True)
    return refs


def _mark_toc_blocks(refs: Sequence[_Ref]) -> None:
    """Flag runs of short page-number-terminated lines as table of contents."""
    looks_like = [
        bool(probe and len(probe) <= TOC_LINE_MAX_CHARS and _TOC_TAIL_RE.search(probe))
        for probe in ((r.text or r.title) for r in refs)
    ]
    start = 0
    while start < len(refs):
        if not looks_like[start]:
            start += 1
            continue
        end = start
        while end < len(refs) and looks_like[end]:
            end += 1
        if end - start >= TOC_RUN_MIN:
            for ref in refs[start:end]:
                ref.is_toc = True
        start = end


def _heading_key(title: str) -> str:
    return " ".join(re.sub(r"[\s.:]+$", "", title or "").split()).casefold()


def _is_plausible_heading(title: str) -> bool:
    if any(pattern.search(title) for pattern in _IMPLAUSIBLE_HEADING_RES):
        return False
    return _heading_key(title) not in _NON_UNIT_HEADINGS


def _reads_like_sentence(title: str) -> bool:
    """Stricter than plausibility: only used to *name* a chapter by the line
    before its first section, where a paragraph opener often sits."""
    words = title.split()
    if title[:1].islower():
        return True
    if any(pattern.search(title) for pattern in _NOT_A_CHAPTER_NAME_RES):
        return True
    if ", " in title and len(words) >= 6 and ":" not in title:
        return True
    return title.endswith(".") and len(words) > 6


def _chapter_match(title: Optional[str]) -> Optional[tuple]:
    matched = match_chapter_heading(title)
    if matched or not title:
        return matched
    lead = _REDUNDANT_NUMBER_RE.match(title)
    if not lead:
        return None
    matched = match_chapter_heading(title[lead.end() :])
    if matched and matched[2] == int(lead.group(1)):
        return matched
    return None


def _mark_toc_heading_runs(refs: Sequence[_Ref], max_gap_chars: int) -> None:
    """Flag runs of structural headings with almost no text between them.

    A table of contents without page numbers slips past `_mark_toc_blocks`:
    CompTIA lists Chapter 1…12 and its appendix on pp. 39–42, and those entries
    outranked the body headings.
    """
    hits = [r for r in refs if not r.is_toc and r.title and _chapter_match(r.title)]
    # A heading is a TOC entry when the next heading follows almost at once;
    # the body heading right after the TOC owns real text and stays out.
    short = [
        i + 1 < len(hits) and _span_mass(refs, hit.index, hits[i + 1].index) < max_gap_chars
        for i, hit in enumerate(hits)
    ]
    start = 0
    while start < len(hits):
        if not short[start]:
            start += 1
            continue
        end = start
        while end < len(hits) and short[end]:
            end += 1
        if end - start >= TOC_HEADING_RUN_MIN:
            for ref in refs[hits[start].index : hits[end - 1].index + 1]:
                ref.is_toc = True
        start = end


def _mark_anchor_eligibility(refs: Iterable[_Ref]) -> None:
    for ref in refs:
        ref.anchor_eligible = bool(
            ref.title
            and len(ref.title) <= ANCHOR_TITLE_MAX_CHARS
            and ref.label not in NON_ANCHOR_LABELS
            and not ref.is_toc
            and not (ref.title == ref.text and len(ref.title) > TITLE_IS_BODY_MIN_CHARS)
            and _is_plausible_heading(ref.title)
        )


def _span_mass(refs: Sequence[_Ref], start: int, end: int) -> int:
    return sum(r.mass for r in refs[start:end] if not r.is_toc)


def _longest_increasing(items: Sequence[Tuple[Any, int, int]]) -> List[Any]:
    """Longest subsequence (in reading order) whose ordinals strictly climb.

    Items are ``(payload, ordinal, weight)``; equal-length chains are broken by
    total weight (text owned), so a stray «Chapter 5, Chapter 6» blurb in the
    front matter loses to the body's «Chapter 1 … Chapter 4» (CompTIA). Greedy
    "drop anything out of order" used to keep whichever came first.
    """
    if not items:
        return []
    score = [(1, weight) for _, _, weight in items]
    prev = [-1] * len(items)
    for i, (_, ordinal, weight) in enumerate(items):
        for j in range(i):
            if items[j][1] < ordinal:
                candidate = (score[j][0] + 1, score[j][1] + weight)
                if candidate > score[i]:
                    score[i], prev[i] = candidate, j
    i = max(range(len(items)), key=lambda k: score[k])
    out = []
    while i >= 0:
        out.append(items[i][0])
        i = prev[i]
    return out[::-1]


def _vocabulary_anchors(refs: Sequence[_Ref], kinds: Iterable[str], total: int = 0) -> List[int]:
    wanted = set(kinds)
    # (ref, match, scope): chapter numbering may restart in every part
    # (Penetration Testing: A Survival Guide has Chapter 1…10 in each module).
    hits: List[Tuple[_Ref, tuple, int]] = []
    scope = 0
    answer_key_zone = False
    position = 0
    for ref in refs:
        if not ref.is_toc:
            position += ref.mass
        if not ref.anchor_eligible:
            continue
        matched = _chapter_match(ref.title)
        if not matched or matched[0] not in wanted:
            continue
        kind = matched[0]
        if kind == "appendix" and total and position < total / 10:
            continue  # front matter mentioning an appendix, not the appendix
        if kind == "appendix" and total and position >= total / 2:
            # Back-matter answer keys restate every chapter heading (CompTIA).
            answer_key_zone = True
        elif answer_key_zone and kind in ("chapter", "part"):
            continue
        if kind == "part":
            scope += 1
        hits.append((ref, matched, scope if kind == "chapter" else 0))
    if not hits:
        return []

    # A heading that also appears in a short table of contents (too short to be
    # caught as a run) shows up twice — keep the occurrence that owns real text.
    indices = [ref.index for ref, _, _ in hits]
    spans = {
        ref.index: _span_mass(
            refs, ref.index, indices[i + 1] if i + 1 < len(indices) else len(refs)
        )
        for i, (ref, _, _) in enumerate(hits)
    }
    best: Dict[tuple, Tuple[_Ref, tuple, int]] = {}
    for ref, matched, sc in hits:
        key = (matched[0], sc, matched[2] if matched[2] is not None else matched[1])
        current = best.get(key)
        if current is None or spans[ref.index] > spans[current[0].index]:
            best[key] = (ref, matched, sc)

    # Ordinals must climb within a kind (and part scope) — a body mention that
    # slipped through the anchor filter shows up as Глава 2 after Глава 7.
    groups: Dict[tuple, List[Tuple[_Ref, int]]] = {}
    picks: List[Tuple[_Ref, tuple]] = []
    for ref, matched, sc in sorted(best.values(), key=lambda t: t[0].index):
        if matched[2] is None:
            picks.append((ref, matched))
        else:
            groups.setdefault((matched[0], sc), []).append(
                ((ref, matched), matched[2], spans[ref.index])
            )
    for items in groups.values():
        picks.extend(_longest_increasing(items))
    picks.sort(key=lambda pair: pair[0].index)

    # Parts are containers: with real chapters found, a part heading only
    # splits its first chapter's unit (or is a lab step named "Part 2: …").
    if sum(1 for _, matched in picks if matched[0] == "chapter") >= 3:
        picks = [pair for pair in picks if pair[1][0] != "part"]
    return [ref.index for ref, _ in picks]


def _section_prefix_anchors(refs: Sequence[_Ref]) -> List[int]:
    """Chapters recovered from multi-level section numbers: 3.1, 3.2 … → chapter 3.

    Several E2E books never print the word "Chapter" in the body (Switching
    Power Supply Design: 17 chapters, the vocabulary tier found 2), yet number
    every section. The chapter is named by the unnumbered line just before its
    first section when there is one.
    """
    runs: List[List[int]] = []  # [chapter_number, first_ref_index]
    for ref in refs:
        if not ref.anchor_eligible:
            continue
        ordinal = parse_section_ordinal(ref.title)
        if not ordinal or len(ordinal) < 2:
            continue
        if runs and runs[-1][0] == ordinal[0]:
            continue
        runs.append([ordinal[0], ref.index])
    if len(runs) < 3:
        return []
    chosen = _longest_increasing([(start, number, 0) for number, start in runs])
    # Per-paper numbering in an edited volume restarts constantly — not chapters.
    if len(chosen) < 3 or len(chosen) < PREFIX_MONOTONIC_MIN * len(runs):
        return []
    return [_prefix_chapter_start(refs, start) for start in chosen]


def _prefix_chapter_start(refs: Sequence[_Ref], first_section: int) -> int:
    for ref in reversed(refs[max(0, first_section - 3) : first_section]):
        if ref.is_toc:
            break
        ordinal = parse_section_ordinal(ref.title)
        if ordinal and len(ordinal) >= 2:
            break  # still inside the previous chapter
        if (
            ref.anchor_eligible
            and not _reads_like_sentence(ref.title)
            and _heading_key(ref.title) not in _FRONT_MATTER_TITLES
        ):
            return ref.index
    return first_section


def repair_cp1251_mojibake(text: str) -> str:
    """«ÌÈÊÐÎÏÐÎÖÅÑÑÎÐ» → «МИКРОПРОЦЕССОР»: a cp1251 text layer read as latin-1."""
    if not re.search(r"[\u00c0-\u00ff]{3,}", text or ""):
        return text
    try:
        fixed = text.encode("latin-1").decode("cp1251")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    letters = [ch for ch in fixed if ch.isalpha()]
    cyrillic = sum(1 for ch in letters if "\u0400" <= ch <= "\u04ff")
    return fixed if letters and cyrillic >= 0.6 * len(letters) else text


def _toc_chapter_names(refs: Sequence[_Ref]) -> Dict[int, str]:
    """Chapter number → name, from the book's own contents list.

    Only lines that state a chapter («Глава 2. Полупроводники. 25», «Chapter 3:
    Information Gathering»), or a bare «2 Name 25» immediately followed by its
    «2.1» — a numbered line on its own could be anything.
    """
    lines: List[str] = []
    for ref in refs:
        if ref.is_toc or _heading_key(ref.title) in _CONTENTS_HEADINGS:
            body = ref.text if len(ref.text) > len(ref.title) else ref.title
            lines.extend(line.strip() for line in body.splitlines() if line.strip())
    names: Dict[int, str] = {}
    for i, line in enumerate(lines):
        entry = repair_cp1251_mojibake(_TOC_PAGE_TAIL_RE.sub("", line).strip(" ."))
        matched = match_chapter_heading(entry)
        if matched and matched[0] == "chapter" and matched[2]:
            if split_chapter_heading(entry)[1]:
                names.setdefault(matched[2], entry)
            continue
        plain = _TOC_PLAIN_CHAPTER_RE.match(entry)
        if plain and i + 1 < len(lines) and lines[i + 1].startswith(f"{plain.group(1)}.1"):
            names.setdefault(int(plain.group(1)), plain.group(2).strip())
    return names


def _end_marker_anchors(refs: Sequence[_Ref]) -> List[int]:
    """Chapters recovered from their closing sections, one cluster per chapter.

    Markers before the end of the contents list are the introduction describing
    the study aids, not a chapter closing.
    """
    toc_end = max((r.index for r in refs if r.is_toc), default=-1)
    hits = [
        r.index
        for r in refs
        if r.index > toc_end and not r.is_toc and _heading_key(r.title) in _END_MARKER_HEADINGS
    ]
    clusters: List[int] = []  # last marker of each cluster
    seen: set = set()  # marker kinds in the current cluster
    for hit in hits:
        key = _heading_key(refs[hit].title)
        if (
            clusters
            and key not in seen
            and _span_mass(refs, clusters[-1], hit) < END_MARKER_CLUSTER_GAP
        ):
            clusters[-1] = hit
            seen.add(key)
        else:
            # A marker seen again is the next chapter closing.
            clusters.append(hit)
            seen = {key}
    if len(clusters) < END_MARKER_MIN_CHAPTERS:
        return []

    def _next_opening(after: int) -> Optional[int]:
        return next(
            (
                r.index
                for r in refs[after + 1 :]
                if r.anchor_eligible
                and _heading_key(r.title) not in _END_MARKER_HEADINGS
                and not _NOT_AN_OPENING_RE.search(r.title)
            ),
            None,
        )

    def _back_matter(after: int) -> Optional[int]:
        # Past the last closing, a lab listing or a question may precede the
        # appendix heading (DOC_012's PowerShell line) — the heading names it.
        for r in refs[after + 1 :]:
            matched = r.anchor_eligible and _chapter_match(r.title)
            if matched and matched[0] in ("appendix", "part"):
                return r.index
        return _next_opening(after)

    anchors = [_next_opening(toc_end)] if toc_end >= 0 else []
    anchors += [_next_opening(last) for last in clusters[:-1]]
    anchors.append(_back_matter(clusters[-1]))
    return sorted({a for a in anchors if a is not None})


def _toc_titles(
    refs: Sequence[_Ref], anchors: Sequence[int], tier: str, toc_names: Dict[int, str]
) -> Dict[int, str]:
    """Anchor index → the contents list's name for that chapter.

    - end_markers: the chapters are in contents order, so the k-th anchor is
      chapter k when there are as many closings as listed chapters;
    - otherwise an anchor that is a subsection («2.1. Введение») stands in for
      a chapter heading OCR lost — its chapter's name is in the contents list.
    """
    if not toc_names:
        return {}
    titles: Dict[int, str] = {}
    ordered = [toc_names[k] for k in sorted(toc_names)]
    if tier == "end_markers" and len(anchors) in (len(ordered), len(ordered) + 1):
        return dict(zip(anchors, ordered))
    for index in anchors:
        ordinal = parse_section_ordinal(refs[index].title)
        if ordinal and len(ordinal) >= 2 and ordinal[0] in toc_names:
            titles[index] = toc_names[ordinal[0]]
    return titles


def _numbered_section_anchors(refs: Sequence[_Ref]) -> List[int]:
    anchors: List[int] = []
    previous: Optional[int] = None
    for ref in refs:
        if not ref.anchor_eligible:
            continue
        ordinal = parse_section_ordinal(ref.title)
        if not ordinal or len(ordinal) != 1:
            continue
        if previous is not None and ordinal[0] <= previous:
            continue
        previous = ordinal[0]
        anchors.append(ref.index)
    return anchors


def _root_child_anchors(refs: Sequence[_Ref], max_units: int, min_unit_chars: int) -> List[int]:
    """Preserve the old behaviour for trees that were already clean."""
    starts = [r.index for r in refs if r.is_root_child and r.anchor_eligible]
    if not starts or len(starts) > max_units:
        return []
    masses = [
        _span_mass(refs, start, starts[i + 1] if i + 1 < len(starts) else len(refs))
        for i, start in enumerate(starts)
    ]
    if not masses or median(masses) < min_unit_chars:
        return []
    return starts


def _mass_segmentation_anchors(refs: Sequence[_Ref], total: int, max_units: int) -> List[int]:
    if total <= 0:
        return [0] if refs else []
    count = max(1, min(max_units, round(total / TARGET_UNIT_CHARS)))
    if count == 1:
        return [0]

    step = total / count
    anchors = [0]
    running = 0
    target = step
    for ref in refs:
        if ref.is_toc:
            continue
        running += ref.mass
        if running >= target and ref.index > anchors[-1]:
            # Prefer a plausible heading at the cut, else cut on mass.
            cut = ref.index
            for candidate in refs[ref.index : min(ref.index + 5, len(refs))]:
                if candidate.anchor_eligible:
                    cut = candidate.index
                    break
            if cut > anchors[-1]:
                anchors.append(cut)
            target += step
            if len(anchors) >= count:
                break
    return anchors


def _unit_title(anchor: _Ref, span: Sequence[_Ref], fallback: str) -> str:
    """The unit's heading, with its name restored when the anchor has none.

    A two-line heading — `Приложение Б.` above `Числа с плавающей точкой` —
    arrives as two nodes, so the anchor carries the label and nothing else and
    §2.2 printed a bare `Phụ lục B.`. The book's own table of contents reads
    `Приложение Б. Числа с плавающей точкой`: the name is in the document, one
    node further on.

    So it is quoted from the next member rather than invented. `anchor_eligible`
    already means "this title is a real heading, not body text thinning promoted"
    — the same question — so junk titles are skipped by reusing it, and a member
    that declares its own structure never lends its name.
    """
    title = (anchor.title or fallback).strip()
    heading, name = split_chapter_heading(title)
    if not heading or name:
        return title
    borrowed = next(
        (
            r.title.strip()
            for r in span[1:]
            if r.anchor_eligible and r.title and match_chapter_heading(r.title) is None
        ),
        "",
    )
    return f"{title.rstrip('. ')}. {borrowed}" if borrowed else title


def _build_units(
    refs: Sequence[_Ref],
    anchors: Sequence[int],
    total: int,
    titles: Optional[Dict[int, str]] = None,
) -> List[_Unit]:
    if not anchors:
        return []

    units: List[_Unit] = []
    for i, start in enumerate(anchors):
        end = anchors[i + 1] if i + 1 < len(anchors) else len(refs)
        span = [r for r in refs[start:end] if not r.is_toc]
        if span:
            units.append(
                _Unit(
                    title=(titles or {}).get(start)
                    or _unit_title(refs[start], span, f"Phần {i + 1}"),
                    refs=span,
                    anchor_page=refs[start].page,
                )
            )

    prologue = [r for r in refs[: anchors[0]] if not r.is_toc]
    if prologue and units:
        prologue_mass = sum(r.mass for r in prologue)
        if total > 0 and prologue_mass >= PROLOGUE_MIN_RATIO * total:
            title = next((r.title for r in prologue if r.anchor_eligible), "") or "Mở đầu"
            units.insert(0, _Unit(title=title, refs=prologue))
        else:
            # Never drop content — thin front matter rides along with unit 1.
            units[0].refs = prologue + units[0].refs
    elif prologue and not units:
        units.append(_Unit(title=prologue[0].title or "Mở đầu", refs=prologue))
    return units


def _absorb_small_units(units: List[_Unit], min_chars: int) -> List[_Unit]:
    """A unit under half a page is a fragment: fold it into its neighbour.

    This is the operational definition of "not fragmented" — and it moves text,
    it never deletes it.
    """
    if len(units) <= 1:
        return units
    merged: List[_Unit] = []
    for unit in units:
        if merged and unit.mass < min_chars:
            merged[-1].refs.extend(unit.refs)
        else:
            merged.append(unit)
    if len(merged) > 1 and merged[0].mass < min_chars:
        merged[1].refs = merged[0].refs + merged[1].refs
        merged.pop(0)
    return merged


def _cap_units(units: List[_Unit], max_units: int) -> List[_Unit]:
    while len(units) > max_units:
        smallest = min(range(len(units)), key=lambda i: units[i].mass)
        left = units[smallest - 1] if smallest > 0 else None
        right = units[smallest + 1] if smallest + 1 < len(units) else None
        if left is None or (right is not None and right.mass < left.mass):
            right.refs = units[smallest].refs + right.refs
        else:
            left.refs.extend(units[smallest].refs)
        units.pop(smallest)
    return units


def select_chapter_units(tree: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return ``(unit_nodes, metadata)`` for digest §2.2.

    ``unit_nodes`` are ordinary tree-node dicts, so every downstream helper
    (gather_node_text, build_stratified_sample, the §2.2 noise gate) consumes
    them unchanged.
    """
    min_unit_chars = settings.main_content_min_unit_chars
    coverage_min = settings.main_content_unit_coverage_min

    meta: Dict[str, Any] = {
        "unit_selection_tier": None,
        "units_before_merge": 0,
        "coverage_ratio": 0.0,
        "median_unit_chars": 0,
        "max_units": effective_max_units(0),
    }

    refs = _flatten_preorder(tree)
    if not refs:
        return [], meta

    _mark_toc_blocks(refs)
    _mark_toc_heading_runs(refs, 2 * min_unit_chars)
    _mark_anchor_eligibility(refs)
    total = sum(r.mass for r in refs if not r.is_toc)
    # The ceiling depends on how much document there actually is, so it can only
    # be known once the tree has been flattened and weighed.
    max_units = effective_max_units(total)
    meta["max_units"] = max_units

    def _coverage(found: List[int]) -> float:
        # Anchors starting deep in the body describe only their own tail — that
        # is a body mention, not the document's structure.
        return (_span_mass(refs, found[0], len(refs)) / total) if total else 1.0

    chosen_tier = "mass_segmentation"
    anchors: List[int] = []
    coverage = 0.0

    # The two strongest signals compete: whichever outlines more chapters wins
    # (a tie keeps the explicit vocabulary).
    strong = []
    for tier, found in (
        ("chapter_vocabulary", _vocabulary_anchors(refs, STRONG_KINDS, total)),
        ("section_prefix", _section_prefix_anchors(refs)),
    ):
        if found and _coverage(found) >= coverage_min:
            strong.append((tier, found))
    closings = _end_marker_anchors(refs)
    best = max((len(found) for _, found in strong), default=0)
    if (
        closings
        and _coverage(closings) >= coverage_min
        and len(closings) >= END_MARKER_MIN_ADVANTAGE * best
    ):
        strong.append(("end_markers", closings))
    if strong:
        chosen_tier, anchors = max(strong, key=lambda pair: len(pair[1]))
        coverage = _coverage(anchors)

    candidates = [
        ("section_vocabulary", lambda: _vocabulary_anchors(refs, {"section"}, total)),
        ("numbered_sections", lambda: _numbered_section_anchors(refs)),
        ("root_children", lambda: _root_child_anchors(refs, max_units, min_unit_chars)),
    ]
    for tier, resolve in candidates if not anchors else ():
        found = resolve()
        if not found:
            continue
        ratio = _coverage(found)
        if ratio < coverage_min:
            continue
        chosen_tier, anchors, coverage = tier, found, ratio
        break

    if not anchors:
        anchors = _mass_segmentation_anchors(refs, total, max_units)
        coverage = (_span_mass(refs, anchors[0], len(refs)) / total) if (total and anchors) else 1.0

    titles = _toc_titles(refs, anchors, chosen_tier, _toc_chapter_names(refs))
    units = _build_units(refs, anchors, total, titles)
    meta["units_before_merge"] = len(units)
    units = _cap_units(_absorb_small_units(units, min_unit_chars), max_units)

    if chosen_tier == "section_prefix":
        for unit in units:
            unit.chapter_ordinal = next(
                (
                    o[0]
                    for o in map(parse_section_ordinal, (r.title for r in unit.refs))
                    if o and len(o) >= 2
                ),
                None,
            )

    meta["unit_selection_tier"] = chosen_tier
    meta["coverage_ratio"] = round(coverage, 4)
    meta["median_unit_chars"] = int(median([u.mass for u in units])) if units else 0
    return [unit.to_node() for unit in units], meta
