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
        → paper_boundaries → article_sections → numbered_sections → root_children
        → mass_segmentation

The last tier always succeeds, so there is no "no structure found" failure mode —
but the tier that fired is reported in the metadata and surfaced as a quality
warning, so a machine-cut digest is never silently passed off as a real outline.
"""

from __future__ import annotations

import re
import unicodedata
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
# Mass-segmentation granularity when no outline is recognised: about a paper
# section. It used to be a book chapter (20k), which cut a 45k-char article into
# two §2.2 entries against the references' six (ND4 run 4); on long documents
# the adaptive cap below decides the count either way.
TARGET_UNIT_CHARS = 8_000

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
# Fewer root children than this and the root_children tier looks one level down.
MIN_ROOT_UNITS = 3

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
# The canonical sections of a research article. A journal paper's outline is
# usually unnumbered — «MỞ ĐẦU … KẾT LUẬN», «Введение … Заключение» — and no
# other tier reads it (ND4 run 4: nine papers got one or two §2.2 entries).
# Compared with whitespace removed: OCR glues spaced capitals into «THẢOLUẬN».
_ARTICLE_SECTION_HEADINGS = frozenset(
    {
        "introduction",
        "background",
        "literaturereview",
        "relatedwork",
        "method",
        "methods",
        "methodology",
        "materialsandmethods",
        "results",
        "findings",
        "resultsanddiscussion",
        "discussion",
        "conclusion",
        "conclusions",
        "mởđầu",
        "đặtvấnđề",
        "giớithiệu",
        "tổngquantàiliệu",
        "tổngquannghiêncứu",
        "cơsởlýthuyết",
        "cơsởlíthuyết",
        "phươngpháp",
        "phươngphápnghiêncứu",
        "vậtliệuvàphươngpháp",
        "kếtquả",
        "kếtquảnghiêncứu",
        "kếtquảvàthảoluận",
        "thảoluận",
        "bànluận",
        "kếtluận",
        "kếtluậnvàkiếnnghị",
        "kếtluậnvàkhuyếnnghị",
        "введение",
        "обзорлитературы",
        "методы",
        "методыисследования",
        "материалыиметоды",
        "результаты",
        "результатыисследования",
        "результатыиобсуждение",
        "обсуждение",
        "обсуждениерезультатов",
        "заключение",
        "выводы",
        "引言",
        "绪论",
        "研究方法",
        "研究结果",
        "结果与讨论",
        "讨论",
        "结论",
        "结语",
    }
)
# The abstract a paper opens with. A volume of papers repeats these, once or twice
# (Vietnamese + English) per paper, a page or so apart.
_ABSTRACT_HEADINGS = frozenset(
    {"abstract", "tómtắt", "tomtat", "аннотация", "резюме", "摘要", "résumé", "resume"}
)
# Abstracts this many pages apart belong to the same paper.
PAPER_ABSTRACT_PAGE_GAP = 1
# Under about two pages of text is a stray abstract, not a paper.
PAPER_MIN_CHARS = 6000
# How much text around a paper's opening is handed over to find its title.
PAPER_HINT_CHARS = 700
PAPER_HINT_AFTER_CHARS = 300
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
    depth: int = 0  # 0 = a root child
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
    # The Part it sits in — {"ordinal", "name"} — when the book has Parts.
    part: Optional[Dict[str, Any]] = None
    # Index of the ref that opens the unit. Not refs[0]: thin front matter is
    # prepended to unit 1.
    anchor_index: Optional[int] = None
    # The title is a placeholder: the unit's name is not in the tree.
    untitled: bool = False
    # The text around the paper's opening, where its title is printed.
    title_hint: str = ""

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
            **({"part": self.part} if self.part else {}),
            **({"untitled": True, "single_paper": True} if self.untitled else {}),
            **({"title_hint": self.title_hint} if self.title_hint else {}),
            # What the summarise stage wrote on the unit's nodes — anchor first. §2.1
            # composes its outline from the same units §2.2 lists.
            **(
                {"summary": summary}
                if (
                    summary := next(
                        (
                            str(r.node.get("summary") or "").strip()
                            for r in self.refs
                            if str(r.node.get("summary") or "").strip()
                        ),
                        "",
                    )
                )
                else {}
            ),
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

    def walk(node: Dict[str, Any], depth: int) -> None:
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
                is_root_child=depth == 0,
                depth=depth,
            )
        )
        for child in _child_nodes(node):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
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


@dataclass
class _PrefixPlan:
    anchors: List[int] = field(default_factory=list)
    # anchor index → chapter number, for every anchor (found or recovered)
    ordinals: Dict[int, int] = field(default_factory=dict)
    # anchor index → the contents list's entry, where the anchor's own title is
    # not the chapter's name (a section, or the Part title page)
    named: Dict[int, str] = field(default_factory=dict)
    # chapter number → contents-list entry, read from the front matter
    names: Dict[int, str] = field(default_factory=dict)


def _section_prefix_anchors(refs: Sequence[_Ref]) -> List[int]:
    return _section_prefix_plan(refs).anchors


def _section_prefix_plan(refs: Sequence[_Ref]) -> _PrefixPlan:
    """Chapters recovered from multi-level section numbers: 3.1, 3.2 … → chapter 3.

    Several E2E books never print the word "Chapter" in the body (Switching
    Power Supply Design: 17 chapters, the vocabulary tier found 2), yet number
    every section. The chapter is named by the unnumbered line just before its
    first section when there is one.

    A chapter with no numbered section (Ballistics 6, 7, 10, 14, 19) leaves no
    run, so it would be folded into the chapter before it. The book's own
    contents list says which chapters exist; the gaps are filled from it.
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
        return _PrefixPlan()
    chosen = _longest_increasing([(start, number, 0) for number, start in runs])
    # Per-paper numbering in an edited volume restarts constantly — not chapters.
    if len(chosen) < 3 or len(chosen) < PREFIX_MONOTONIC_MIN * len(runs):
        return _PrefixPlan()

    number_of = {start: number for number, start in runs}
    plan = _PrefixPlan()
    for start in chosen:
        anchor = _prefix_chapter_start(refs, start)
        plan.ordinals[anchor] = number_of[start]

    first_anchor = min(plan.ordinals)
    plan.names = _chapter_names_from_lines(
        [line for ref in refs[:first_anchor] for line in _ref_lines(ref)]
    )
    if plan.names:
        _restore_chapter_headings(refs, plan)
        _fill_chapter_gaps(refs, plan)
    plan.anchors = sorted(plan.ordinals)
    return plan


def _entry_key(entry: str) -> str:
    return _heading_key(split_chapter_heading(entry)[1])


def _heading_in(refs: Sequence[_Ref], lo: int, hi: int, entry: str) -> Optional[int]:
    """The heading in (lo, hi) that carries the contents list's name for a chapter."""
    key = _entry_key(entry)
    if not key:
        return None
    return next(
        (
            r.index
            for r in refs[lo + 1 : hi]
            if r.anchor_eligible and not r.is_toc and _heading_key(r.title) == key
        ),
        None,
    )


def _restore_chapter_headings(refs: Sequence[_Ref], plan: _PrefixPlan) -> None:
    """A chapter that first shows as a section («11.1 …»), as its Part's title page or as
    a bare number («1») gets its real name from the contents list, and its heading when
    one precedes it."""
    ordered = sorted(plan.ordinals)
    for position, anchor in enumerate(ordered):
        number = plan.ordinals[anchor]
        entry = plan.names.get(number)
        if not entry:
            continue
        section = parse_section_ordinal(refs[anchor].title)
        after_part = (
            anchor > 0 and (match_chapter_heading(refs[anchor - 1].title) or ("",))[0] == "part"
        )
        bare_number = refs[anchor].title.strip().isdigit()
        if not (section and len(section) >= 2) and not after_part and not bare_number:
            continue
        previous = ordered[position - 1] if position else -1
        heading = _heading_in(refs, previous, anchor, entry) if section else None
        if heading is not None:
            plan.ordinals[heading] = plan.ordinals.pop(anchor)
            anchor = heading
        plan.named[anchor] = entry


def _fill_chapter_gaps(refs: Sequence[_Ref], plan: _PrefixPlan) -> None:
    """Chapters the contents list names but no numbered section marks.

    Between its neighbours a chapter is found by its heading, else — when it opens
    a Part, whose title page holds its text — by the Part heading that is there.
    """
    for number in sorted(plan.names):
        if number in plan.ordinals.values():
            continue
        before = [a for a, n in plan.ordinals.items() if n < number]
        if not before:
            continue  # nothing anchors the lower edge of the window
        lo = max(before)
        hi = min((a for a, n in plan.ordinals.items() if n > number), default=len(refs))
        found = _heading_in(refs, lo, hi, plan.names[number])
        if found is None:
            window = [r for r in refs[lo + 1 : hi] if not r.is_toc]
            for position, ref in enumerate(window):
                if (match_chapter_heading(ref.title) or ("",))[0] != "part":
                    continue
                found = next(
                    (
                        r.index
                        for r in window[position + 1 :]
                        if r.anchor_eligible and r.index not in plan.ordinals
                    ),
                    None,
                )
                break
        if found is not None and found not in plan.ordinals:
            plan.ordinals[found] = number
            plan.named[found] = plan.names[number]


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
            lines.extend(_ref_lines(ref))
    return _chapter_names_from_lines(lines)


def _ref_lines(ref: _Ref) -> List[str]:
    body = ref.text if len(ref.text) > len(ref.title) else ref.title
    return [line.strip() for line in body.splitlines() if line.strip()]


def _chapter_names_from_lines(lines: Sequence[str]) -> Dict[int, str]:
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


def _article_section_anchors(refs: Sequence[_Ref], min_unit_chars: int) -> List[int]:
    """Each canonical article section, first occurrence only — a bilingual
    paper restates its outline in the second language's abstract.

    Only sections that own real text count towards the outline: two-column
    reading order can drop a heading far from its body (DOC_235: «KẾT QUẢ»
    with 141 chars, its neighbour swallowing 40k)."""
    anchors: List[int] = []
    seen: set = set()
    for ref in refs:
        if not ref.anchor_eligible:
            continue
        key = re.sub(r"[\s.:]+", "", unicodedata.normalize("NFC", ref.title).casefold())
        if key in _ARTICLE_SECTION_HEADINGS and key not in seen:
            seen.add(key)
            anchors.append(ref.index)
    owners = sum(
        _span_mass(refs, start, anchors[i + 1] if i + 1 < len(anchors) else len(refs))
        >= min_unit_chars
        for i, start in enumerate(anchors)
    )
    return anchors if owners >= MIN_ROOT_UNITS else []


def _paper_boundary_plan(
    refs: Sequence[_Ref], min_unit_chars: int
) -> Tuple[List[int], Dict[int, int]]:
    """Where each paper of a volume opens, and where its title may be found.

    A paper opens at its abstract in the language the volume itself starts with: one
    template prints the Vietnamese abstract first and the English one last, so the
    English abstract of paper k sits right before the Vietnamese one of k+1 and belongs
    to k. The second return value maps each anchor to the first abstract of the run it
    belongs to: the paper's title is printed somewhere inside that window, and is often
    no tree node at all, so the unit is marked untitled and named from its text.

    One article has a single run and falls through to `article_sections`.
    """
    markers = []
    for ref in refs:
        if not ref.anchor_eligible:
            continue
        key = re.sub(r"[\s.:]+", "", unicodedata.normalize("NFC", ref.title).casefold())
        if key in _ABSTRACT_HEADINGS:
            markers.append((ref, key))
    if not markers:
        return [], {}

    opening_key = markers[0][1]
    anchors: List[int] = []
    windows: Dict[int, int] = {}
    last_page: Optional[int] = None
    last_opening_page: Optional[int] = None
    run_start = markers[0][0].index
    for ref, key in markers:
        if not (
            last_page is not None
            and ref.page is not None
            and ref.page - last_page <= PAPER_ABSTRACT_PAGE_GAP
        ):
            run_start = ref.index
        last_page = ref.page
        if key != opening_key:
            continue
        repeated = (
            last_opening_page is not None
            and ref.page is not None
            and ref.page - last_opening_page <= PAPER_ABSTRACT_PAGE_GAP
        )
        last_opening_page = ref.page
        if not repeated:
            anchors.append(ref.index)
            windows[ref.index] = run_start

    # An abstract with almost nothing after it opens nothing: a paper is pages long.
    anchors = [
        start
        for i, start in enumerate(anchors)
        if _span_mass(refs, start, anchors[i + 1] if i + 1 < len(anchors) else len(refs))
        >= PAPER_MIN_CHARS
    ]
    owners = sum(
        _span_mass(refs, start, anchors[i + 1] if i + 1 < len(anchors) else len(refs))
        >= min_unit_chars
        for i, start in enumerate(anchors)
    )
    if owners < MIN_ROOT_UNITS:
        return [], {}
    return anchors, {a: windows[a] for a in anchors}


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
    # «1 Trường Đại học…», «2 Trường…» are affiliations, not an outline.
    return anchors if len(anchors) >= MIN_ROOT_UNITS else []


def _root_child_anchors(
    refs: Sequence[_Ref], max_units: int, min_unit_chars: int, total: int = 0
) -> List[int]:
    """Preserve the old behaviour for trees that were already clean.

    A journal article's tree is usually one root — the paper's title — with the
    sections hanging under it. Taken alone that root made the whole article ONE
    §2.2 entry (11 of the 52 Nội dung 4 articles, up to 216k chars in 5–8
    sentences). With fewer than MIN_ROOT_UNITS roots, their direct children
    anchor too, so the article is outlined by its own sections. When even that
    leaves a long document with too few units (layout found no headings, the
    text sits in the title node), the tier declines and mass segmentation cuts it.
    """
    starts = [r.index for r in refs if r.is_root_child and r.anchor_eligible]
    if 0 < len(starts) < MIN_ROOT_UNITS:
        # A root that is only the title wraps the units rather than being one;
        # counted, it pushed 百戰奇略's ten 卷 over a cap of ten (ND4 run 5).
        starts = [
            r.index
            for r in refs
            if r.anchor_eligible
            and (r.depth == 1 or (r.depth == 0 and r.text not in ("", r.title)))
        ]
        if len(starts) < MIN_ROOT_UNITS and total >= MIN_ROOT_UNITS * TARGET_UNIT_CHARS:
            return []
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


_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


def _part_number(label: str, fallback: int) -> int:
    label = (label or "").strip().upper()
    if label.isdigit():
        return int(label)
    if label and set(label) <= set(_ROMAN):
        total = 0
        for ch, nxt in zip(label, label[1:] + " "):
            value = _ROMAN[ch]
            total += -value if _ROMAN.get(nxt, 0) > value else value
        return total
    return fallback


def _assign_parts(refs: Sequence[_Ref], units: Sequence[_Unit]) -> None:
    """Tag each unit with the Part heading above it (a Part names the span up to the next).

    The name is the Part heading's own words, or — as in Ballistics, where «Part II» and
    «Exterior Ballistics» are two nodes — the title page node right after it.
    """
    current: Optional[Dict[str, Any]] = None
    at: Dict[int, Optional[Dict[str, Any]]] = {}
    for ref in refs:
        if ref.is_toc:
            continue
        matched = match_chapter_heading(ref.title)
        if matched and matched[0] == "part":
            name = split_chapter_heading(ref.title)[1]
            if not name:
                following = next(
                    (
                        r
                        for r in refs[ref.index + 1 : ref.index + 3]
                        if r.anchor_eligible and not r.is_toc
                    ),
                    None,
                )
                name = following.title if following else ""
            current = {"ordinal": _part_number(matched[1], len(at) + 1), "name": name.strip()}
        at[ref.index] = current
    for unit in units:
        opening = unit.anchor_index if unit.anchor_index is not None else unit.refs[0].index
        unit.part = at.get(opening)


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
                    anchor_index=start,
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

    paper_plans: List[Tuple[List[int], Dict[int, int]]] = []

    def paper_plan() -> Tuple[List[int], Dict[int, int]]:
        if not paper_plans:
            paper_plans.append(_paper_boundary_plan(refs, min_unit_chars))
        return paper_plans[0]

    candidates = [
        ("section_vocabulary", lambda: _vocabulary_anchors(refs, {"section"}, total)),
        ("paper_boundaries", lambda: paper_plan()[0]),
        ("article_sections", lambda: _article_section_anchors(refs, min_unit_chars)),
        ("numbered_sections", lambda: _numbered_section_anchors(refs)),
        ("root_children", lambda: _root_child_anchors(refs, max_units, min_unit_chars, total)),
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

    toc_names = _toc_chapter_names(refs)
    plan = _PrefixPlan()
    if chosen_tier == "section_prefix":
        plan = _section_prefix_plan(refs)
        anchors = plan.anchors
        toc_names = {**plan.names, **toc_names}
    titles = _toc_titles(refs, anchors, chosen_tier, toc_names)
    if chosen_tier == "paper_boundaries":
        titles = {a: f"Bài báo {i}" for i, a in enumerate(anchors, start=1)}
    titles.update({a: entry for a, entry in plan.named.items() if a in anchors})
    units = _build_units(refs, anchors, total, titles)
    meta["units_before_merge"] = len(units)
    units = _cap_units(_absorb_small_units(units, min_unit_chars), max_units)

    _assign_parts(refs, units)
    for unit in units:
        unit.untitled = chosen_tier == "paper_boundaries"
        if unit.untitled:
            window_start = paper_plan()[1].get(unit.anchor_index, unit.anchor_index)
            # The title is printed just before the abstract it opens; what follows
            # says what the paper is about.
            anchor = unit.anchor_index or 0

            def _text(span: Sequence[_Ref]) -> str:
                return "\n".join((r.text or r.title).strip() for r in span)

            unit.title_hint = (
                _text(refs[window_start : anchor + 1])[-PAPER_HINT_CHARS:]
                + "\n"
                + _text(refs[anchor + 1 : anchor + 3])[:PAPER_HINT_AFTER_CHARS]
            )
    if chosen_tier == "section_prefix":
        for unit in units:
            unit.chapter_ordinal = plan.ordinals.get(unit.anchor_index) or next(
                (
                    o[0]
                    for o in map(parse_section_ordinal, (r.title for r in unit.refs))
                    if o and len(o) >= 2
                ),
                None,
            )

    if plan.names:
        # What the book's own contents list promises — the quality report compares
        # it with the chapters §2.2 actually carries.
        meta["contents_chapters"] = sorted(plan.names)
    meta["unit_selection_tier"] = chosen_tier
    meta["coverage_ratio"] = round(coverage, 4)
    meta["median_unit_chars"] = int(median([u.mass for u in units])) if units else 0
    return [unit.to_node() for unit in units], meta
