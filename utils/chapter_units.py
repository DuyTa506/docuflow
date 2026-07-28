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

    chapter_vocabulary → section_vocabulary → numbered_sections
        → root_children → mass_segmentation

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
from core.spatial.zone_classifier import match_chapter_heading, parse_section_ordinal

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
                title=str(node.get("title") or "").strip(),
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


def _mark_anchor_eligibility(refs: Iterable[_Ref]) -> None:
    for ref in refs:
        ref.anchor_eligible = bool(
            ref.title
            and len(ref.title) <= ANCHOR_TITLE_MAX_CHARS
            and ref.label not in NON_ANCHOR_LABELS
            and not ref.is_toc
            and not (ref.title == ref.text and len(ref.title) > TITLE_IS_BODY_MIN_CHARS)
        )


def _span_mass(refs: Sequence[_Ref], start: int, end: int) -> int:
    return sum(r.mass for r in refs[start:end] if not r.is_toc)


def _drop_non_monotonic(picks: List[Tuple[_Ref, tuple]]) -> List[Tuple[_Ref, tuple]]:
    """Ordinals must climb within a kind — a body mention that slipped through
    the anchor filter shows up as an out-of-order Глава 2 after Глава 7."""
    last: Dict[str, int] = {}
    kept = []
    for ref, matched in picks:
        kind, _, ordinal = matched
        if ordinal is None:
            kept.append((ref, matched))
            continue
        previous = last.get(kind)
        if previous is not None and ordinal <= previous:
            continue
        last[kind] = ordinal
        kept.append((ref, matched))
    return kept


def _vocabulary_anchors(refs: Sequence[_Ref], kinds: Iterable[str]) -> List[int]:
    wanted = set(kinds)
    hits: List[Tuple[_Ref, tuple]] = []
    for ref in refs:
        if not ref.anchor_eligible:
            continue
        matched = match_chapter_heading(ref.title)
        if matched and matched[0] in wanted:
            hits.append((ref, matched))
    if not hits:
        return []

    # A heading that also appears in a short table of contents (too short to be
    # caught as a run) shows up twice — keep the occurrence that owns real text.
    indices = [ref.index for ref, _ in hits]
    spans = {
        ref.index: _span_mass(
            refs, ref.index, indices[i + 1] if i + 1 < len(indices) else len(refs)
        )
        for i, (ref, _) in enumerate(hits)
    }
    best: Dict[tuple, Tuple[_Ref, tuple]] = {}
    for ref, matched in hits:
        key = (matched[0], matched[2] if matched[2] is not None else matched[1])
        current = best.get(key)
        if current is None or spans[ref.index] > spans[current[0].index]:
            best[key] = (ref, matched)

    picks = sorted(best.values(), key=lambda pair: pair[0].index)
    return [ref.index for ref, _ in _drop_non_monotonic(picks)]


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
    starts = [r.index for r in refs if r.is_root_child]
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


def _build_units(refs: Sequence[_Ref], anchors: Sequence[int], total: int) -> List[_Unit]:
    if not anchors:
        return []

    units: List[_Unit] = []
    for i, start in enumerate(anchors):
        end = anchors[i + 1] if i + 1 < len(anchors) else len(refs)
        span = [r for r in refs[start:end] if not r.is_toc]
        if span:
            units.append(
                _Unit(
                    title=refs[start].title or f"Phần {i + 1}",
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
    _mark_anchor_eligibility(refs)
    total = sum(r.mass for r in refs if not r.is_toc)
    # The ceiling depends on how much document there actually is, so it can only
    # be known once the tree has been flattened and weighed.
    max_units = effective_max_units(total)
    meta["max_units"] = max_units

    candidates = [
        ("chapter_vocabulary", lambda: _vocabulary_anchors(refs, STRONG_KINDS)),
        ("section_vocabulary", lambda: _vocabulary_anchors(refs, {"section"})),
        ("numbered_sections", lambda: _numbered_section_anchors(refs)),
        ("root_children", lambda: _root_child_anchors(refs, max_units, min_unit_chars)),
    ]

    chosen_tier = "mass_segmentation"
    anchors: List[int] = []
    coverage = 0.0
    for tier, resolve in candidates:
        found = resolve()
        if not found:
            continue
        # Anchors starting deep in the body describe only their own tail — that
        # is a body mention, not the document's structure.
        ratio = (_span_mass(refs, found[0], len(refs)) / total) if total else 1.0
        if ratio < coverage_min:
            continue
        chosen_tier, anchors, coverage = tier, found, ratio
        break

    if not anchors:
        anchors = _mass_segmentation_anchors(refs, total, max_units)
        coverage = (_span_mass(refs, anchors[0], len(refs)) / total) if (total and anchors) else 1.0

    units = _build_units(refs, anchors, total)
    meta["units_before_merge"] = len(units)
    units = _cap_units(_absorb_small_units(units, min_unit_chars), max_units)

    meta["unit_selection_tier"] = chosen_tier
    meta["coverage_ratio"] = round(coverage, 4)
    meta["median_unit_chars"] = int(median([u.mass for u in units])) if units else 0
    return [unit.to_node() for unit in units], meta
