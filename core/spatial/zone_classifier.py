"""
Zone Classifier Module

Classifies layout elements into functional zones using heuristic rules.
Zones include: title_block, main_text, figure, table, caption, equation,
header, footer, footnote, etc.

This is a heuristic-only implementation. Can be extended with LLM fallback.
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


class ZoneType(Enum):
    """Enumeration of zone types for document layout."""

    TITLE_BLOCK = "title_block"
    AUTHOR_BLOCK = "author_block"
    ABSTRACT = "abstract"
    SECTION_HEADING = "section_heading"
    MAIN_TEXT = "main_text"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    EQUATION = "equation"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    SIDEBAR = "sidebar"
    UNKNOWN = "unknown"


# Zone priority for reading order
ZONE_PRIORITY = {
    ZoneType.TITLE_BLOCK: 0,
    ZoneType.AUTHOR_BLOCK: 1,
    ZoneType.ABSTRACT: 2,
    ZoneType.SECTION_HEADING: 3,
    ZoneType.MAIN_TEXT: 4,
    ZoneType.FIGURE: 5,
    ZoneType.TABLE: 5,
    ZoneType.CAPTION: 6,
    ZoneType.EQUATION: 4,
    ZoneType.FOOTNOTE: 8,
    ZoneType.HEADER: 9,
    ZoneType.FOOTER: 10,
    ZoneType.PAGE_NUMBER: 10,
    ZoneType.SIDEBAR: 7,
    ZoneType.UNKNOWN: 5,
}


# Caption patterns (regex) - Updated to handle DeepSeek HTML-style output
# Example: <center>Figure 6. Picture of DJI mini2. </center>
CAPTION_PATTERNS = [
    # Standard patterns
    r"^(Figure|Fig\.?)\s*\d+",
    r"^(Table|Tab\.?)\s*\d+",
    r"^(Hình)\s*\d+",  # Vietnamese
    r"^(Bảng)\s*\d+",  # Vietnamese
    r"^(Image|Img\.?)\s*\d+",
    r"^(Chart|Graph)\s*\d+",
    r"^\[\d+\]",  # Reference style
    # HTML-wrapped patterns (DeepSeek output)
    r"^<center>\s*(Figure|Fig\.?)\s*\d+",
    r"^<center>\s*(Table|Tab\.?)\s*\d+",
    r"^<center>\s*(Hình)\s*\d+",
    r"^<center>\s*(Bảng)\s*\d+",
    r"^<center>\s*(Image|Img\.?)\s*\d+",
]


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text for matching.

    Handles DeepSeek output like: <center>Figure 6. Description</center>

    Args:
        text: Text that may contain HTML tags

    Returns:
        Text with HTML tags stripped
    """
    if not text:
        return ""

    # Remove common HTML tags
    cleaned = re.sub(r"</?center>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?b>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?i>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?strong>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?em>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<br\s*/?>", " ", cleaned, flags=re.IGNORECASE)

    # Generic tag removal as fallback
    cleaned = re.sub(r"<[^>]+>", "", cleaned)

    return cleaned.strip()


# ── Structural heading vocabulary ───────────────────────────────────
# Loaded from config/chapter_vocab.json so a new language is a data change, not
# a code change. Used both for zone classification (reading order) and, via
# match_chapter_heading, for picking digest §2.2 units out of a noisy tree.
_VOCAB_PATH = Path(__file__).resolve().parents[2] / "config" / "chapter_vocab.json"

# Kept in code so the module still works if the config file is absent.
_FALLBACK_VOCAB = {
    "en": {
        "keywords": {
            "chapter": ["Chapter"],
            "part": ["Part"],
            "section": ["Section"],
            "appendix": ["Appendix"],
        }
    },
    "vi": {
        "keywords": {
            "chapter": ["Chương"],
            "part": ["Phần"],
            "section": ["Mục"],
            "appendix": ["Phụ lục"],
        }
    },
    "ru": {
        "keywords": {
            "chapter": ["Глава"],
            "part": ["Часть"],
            "section": ["Раздел"],
            "appendix": ["Приложение"],
        }
    },
}


def _load_chapter_vocab() -> Dict[str, dict]:
    if _VOCAB_PATH.is_file():
        try:
            with open(_VOCAB_PATH, encoding="utf-8") as f:
                data = json.load(f)
            vocab = {k: v for k, v in data.items() if not k.startswith("_")}
            if vocab:
                return vocab
        except (OSError, ValueError) as exc:  # malformed config must not break extraction
            logging.getLogger(__name__).warning(
                "chapter_vocab.json unreadable (%s) — falling back to built-in vocabulary", exc
            )
    return _FALLBACK_VOCAB


CHAPTER_VOCAB = _load_chapter_vocab()

# Kinds that are still a heading without an ordinal: books routinely carry a
# single unnumbered "Приложение" / "Phụ lục".
_ORDINAL_OPTIONAL_KINDS = frozenset({"appendix"})
# Arabic, Roman, or a single Latin/Cyrillic letter.
_ORDINAL_RE = r"(?:\d{1,3}|[IVXLC]{1,6}|[A-Z]|[А-Я])"
# Beyond this a "heading" is a paragraph that happens to open with the word.
CHAPTER_HEADING_MAX_CHARS = 150

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_CJK_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def chapter_keywords(languages: Optional[Iterable[str]] = None) -> Dict[str, List[str]]:
    """Merge the per-language prefix vocabulary into ``{kind: [words]}``."""
    merged: Dict[str, List[str]] = {}
    wanted = set(languages) if languages else None
    for lang, entry in CHAPTER_VOCAB.items():
        if wanted is not None and lang not in wanted:
            continue
        for kind, words in (entry.get("keywords") or {}).items():
            merged.setdefault(kind, []).extend(words)
    return merged


CHAPTER_KEYWORDS = chapter_keywords()


@lru_cache(maxsize=8)
def _chapter_heading_matchers(
    languages: Optional[Tuple[str, ...]],
) -> Tuple[Tuple[str, "re.Pattern"], ...]:
    matchers: List[Tuple[str, re.Pattern]] = []
    for kind, words in chapter_keywords(languages).items():
        if not words:
            continue
        # Longest first so "Partie" wins over "Part".
        alternation = "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))
        matchers.append(
            (
                kind,
                re.compile(
                    rf"^\s*(?:{alternation})\s*({_ORDINAL_RE})?\s*(?:[.:—–\-]|\s|$)",
                    re.IGNORECASE,
                ),
            )
        )
    # Languages where the ordinal is infixed/suffixed (第1章) ship full regexes.
    wanted = set(languages) if languages else None
    for lang, entry in CHAPTER_VOCAB.items():
        if wanted is not None and lang not in wanted:
            continue
        for kind, patterns in (entry.get("patterns") or {}).items():
            for pattern in patterns:
                try:
                    matchers.append((kind, re.compile(pattern, re.IGNORECASE)))
                except re.error as exc:
                    logging.getLogger(__name__).warning(
                        "chapter_vocab.json: bad regex for %s/%s (%s)", lang, kind, exc
                    )
    return tuple(matchers)


_SECTION_ORDINAL_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3})*)\.?\s+\S")

# Section numbering patterns
SECTION_PATTERNS = [
    r"^\d+\.(\d+\.)*\s+\S",  # 1. Introduction, 1.2.3 Methods
    r"^[A-Z]+\.(\d+\.)*\s+\S",  # A.1 Appendix
] + [
    rf"^({'|'.join(re.escape(w) for w in words)})\s+\S"
    for words in CHAPTER_KEYWORDS.values()
    if words
]


def _roman_to_int(text: str) -> Optional[int]:
    total = 0
    prev = 0
    for ch in reversed(text.upper()):
        value = _ROMAN_VALUES.get(ch)
        if value is None:
            return None
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return total or None


def _cjk_to_int(raw: str) -> Optional[int]:
    """一 / 十二 / 二十三 → 1 / 12 / 23. Enough for chapter ordinals."""
    if "十" not in raw:
        total = 0
        for ch in raw:
            digit = _CJK_DIGITS.get(ch)
            if digit is None:
                return None
            total = total * 10 + digit
        return total or None
    head, _, tail = raw.partition("十")
    tens = _CJK_DIGITS.get(head, 1) if head else 1
    units = _CJK_DIGITS.get(tail, 0) if tail else 0
    return tens * 10 + units


def _ordinal_to_int(raw: str) -> Optional[int]:
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw[0] in _CJK_DIGITS or raw[0] == "十":
        return _cjk_to_int(raw)
    if len(raw) == 1 and raw.isalpha():
        # Latin and Cyrillic alphabets both start their ordinal run at 1.
        base = "A" if raw.isascii() else "А"
        return ord(raw.upper()) - ord(base) + 1
    return _roman_to_int(raw)


def match_chapter_heading(
    text: Optional[str], languages: Optional[Iterable[str]] = None
) -> Optional[Tuple[str, Optional[str], Optional[int]]]:
    """Recognise a structural heading: ``("chapter", "1", 1)``.

    Anchored and length-capped so a body sentence opening with "Глава 2 …" or a
    paragraph beginning "Chapter 3 introduced …" is not mistaken for a heading.

    ``languages`` restricts the vocabulary to those config keys; by default every
    language in config/chapter_vocab.json is tried, which is safe because the
    match must start the string and the string must be heading-short.
    """
    if not text:
        return None
    candidate = text.strip()
    if not candidate or len(candidate) > CHAPTER_HEADING_MAX_CHARS:
        return None

    key = tuple(sorted(languages)) if languages else None
    for kind, pattern in _chapter_heading_matchers(key):
        matched = pattern.match(candidate)
        if not matched:
            continue
        ordinal = matched.group(1) if matched.re.groups else None
        if not ordinal and kind not in _ORDINAL_OPTIONAL_KINDS:
            continue
        return kind, ordinal, _ordinal_to_int(ordinal) if ordinal else None
    return None


def split_chapter_heading(
    text: Optional[str], languages: Optional[Iterable[str]] = None
) -> Tuple[Optional[Tuple[str, Optional[str], Optional[int]]], str]:
    """Split ``"Глава 1. Введение"`` into its structural prefix and its name.

    The digest's §2.2 line is ``Chương 1. Giới thiệu (Введение).`` — the label is
    rendered in Vietnamese, so the source-language label must come off the title
    or it prints twice (``Chương 1. Глава 1. Введение.``, observed on N4.11.160).

    Returns ``(match_or_None, remainder)``. The remainder is empty when the title
    is nothing but a label (``"Приложение Б."`` — observed on N4.11.160): that unit
    genuinely has no name, and the caller renders the label from the returned
    kind/ordinal, so keeping the source label as a "name" printed it twice.
    """
    if not text:
        return None, ""
    candidate = text.strip()
    matched = match_chapter_heading(candidate, languages)
    if matched is None:
        return None, candidate

    key = tuple(sorted(languages)) if languages else None
    for kind, pattern in _chapter_heading_matchers(key):
        found = pattern.match(candidate)
        if not found or kind != matched[0]:
            continue
        return matched, candidate[found.end() :].lstrip(" .:-–—\t")
    return matched, candidate


def parse_section_ordinal(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse a numbered-section prefix into its components.

    Accepts both ``"16 Title"`` and ``"16. Title"`` — the repo's two existing
    numbering matchers each rejected one of those forms.
    """
    if not text:
        return None
    matched = _SECTION_ORDINAL_RE.match(text.strip())
    if not matched:
        return None
    return tuple(int(part) for part in matched.group(1).split("."))


# Page number patterns
PAGE_NUMBER_PATTERNS = [
    r"^\d{1,4}$",  # Standalone numbers
    r"^-\s*\d+\s*-$",  # - 5 -
    r"^page\s*\d+",  # Page 5
    r"^trang\s*\d+",  # Vietnamese: Trang 5
]


@dataclass
class ZoneClassification:
    """Result of zone classification for an element."""

    zone: ZoneType
    confidence: float  # 0.0 to 1.0
    method: str  # 'heuristic', 'llm', or 'fallback'
    features: Dict = None  # Features used for classification


def classify_by_label(
    element: Dict, label_to_zone: Optional[Dict[str, ZoneType]] = None
) -> Optional[ZoneClassification]:
    """
    Classify zone based on OCR grounding label.

    Args:
        element: Layout element with 'label' field
        label_to_zone: Optional custom mapping

    Returns:
        ZoneClassification or None if label not definitive
    """
    if label_to_zone is None:
        label_to_zone = {
            "title": ZoneType.TITLE_BLOCK,
            "sub_title": ZoneType.SECTION_HEADING,
            "subtitle": ZoneType.SECTION_HEADING,
            "heading": ZoneType.SECTION_HEADING,
            "header": ZoneType.HEADER,
            "figure": ZoneType.FIGURE,
            "table": ZoneType.TABLE,
            "equation": ZoneType.EQUATION,
            "formula": ZoneType.EQUATION,
            "isolate_formula": ZoneType.EQUATION,
            "math": ZoneType.EQUATION,
            "image": ZoneType.FIGURE,
            "caption": ZoneType.CAPTION,
            "footnote": ZoneType.FOOTNOTE,
            "footer": ZoneType.FOOTER,
            "page_number": ZoneType.PAGE_NUMBER,
        }

    label = element.get("label", "").lower().strip()

    if label in label_to_zone:
        from config.spatial_config import spatial_config

        return ZoneClassification(
            zone=label_to_zone[label],
            confidence=spatial_config.label_confidence,
            method="heuristic_label",
        )

    return None


def classify_by_position(element: Dict, page_dims: Dict[str, int]) -> Optional[ZoneClassification]:
    """
    Classify zone based on bbox position on page.

    Args:
        element: Layout element with bbox
        page_dims: Page dimensions {'width': int, 'height': int}

    Returns:
        ZoneClassification or None if position not definitive
    """
    page_height = page_dims.get("height", 1000)
    page_width = page_dims.get("width", 800)

    y1 = element.get("bbox_y1", element.get("y1", 0))
    y2 = element.get("bbox_y2", element.get("y2", 0))
    x1 = element.get("bbox_x1", element.get("x1", 0))
    x2 = element.get("bbox_x2", element.get("x2", 0))

    # Relative positions
    rel_y1 = y1 / page_height if page_height > 0 else 0
    rel_y2 = y2 / page_height if page_height > 0 else 0
    rel_x1 = x1 / page_width if page_width > 0 else 0
    rel_x2 = x2 / page_width if page_width > 0 else 0

    # Element dimensions
    elem_width = x2 - x1
    elem_height = y2 - y1
    rel_width = elem_width / page_width if page_width > 0 else 0
    rel_height = elem_height / page_height if page_height > 0 else 0

    # Page number: bottom, centered, very small
    if rel_y1 > 0.92 and rel_height < 0.03 and 0.4 < (rel_x1 + rel_x2) / 2 < 0.6:
        from config.spatial_config import spatial_config

        return ZoneClassification(
            zone=ZoneType.PAGE_NUMBER,
            confidence=spatial_config.position_confidence,
            method="heuristic_position",
            features={"rel_y1": rel_y1, "rel_height": rel_height},
        )

    # Footer zone: bottom of page
    if rel_y1 > 0.9 and rel_height < 0.08:
        return ZoneClassification(
            zone=ZoneType.FOOTER,
            confidence=0.7,
            method="heuristic_position",
            features={"rel_y1": rel_y1},
        )

    # Header zone: top of page
    if rel_y2 < 0.1 and rel_height < 0.08:
        return ZoneClassification(
            zone=ZoneType.HEADER,
            confidence=0.7,
            method="heuristic_position",
            features={"rel_y2": rel_y2},
        )

    # Footnote: bottom, smaller text (based on height)
    if rel_y1 > 0.85 and rel_height < 0.12:
        return ZoneClassification(
            zone=ZoneType.FOOTNOTE,
            confidence=0.6,
            method="heuristic_position",
            features={"rel_y1": rel_y1},
        )

    # Title block: very top, large width
    if rel_y1 < 0.15 and rel_width > 0.5:
        # Could be title, but wait for text pattern check
        pass

    return None


def classify_by_text_pattern(element: Dict) -> Optional[ZoneClassification]:
    """
    Classify zone based on text content patterns.

    Args:
        element: Layout element with text content

    Returns:
        ZoneClassification or None if pattern not matched
    """
    raw_text = element.get("text_content", element.get("text", "")).strip()

    if not raw_text:
        return None

    # Strip HTML tags for matching (handles DeepSeek output)
    text = strip_html_tags(raw_text)

    # Caption patterns (Figure 1, Table 2, etc.)
    # Try both raw (with HTML) and stripped text
    from config.spatial_config import spatial_config

    for pattern in CAPTION_PATTERNS:
        if re.match(pattern, raw_text, re.IGNORECASE) or re.match(pattern, text, re.IGNORECASE):
            return ZoneClassification(
                zone=ZoneType.CAPTION,
                confidence=spatial_config.pattern_confidence,
                method="heuristic_pattern",
                features={"pattern": pattern, "html_stripped": raw_text != text},
            )

    # Page number patterns
    for pattern in PAGE_NUMBER_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return ZoneClassification(
                zone=ZoneType.PAGE_NUMBER,
                confidence=spatial_config.position_confidence,
                method="heuristic_pattern",
                features={"pattern": pattern},
            )

    # Section heading patterns (only if short text)
    if len(text) < 200:  # Headings usually short
        for pattern in SECTION_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return ZoneClassification(
                    zone=ZoneType.SECTION_HEADING,
                    confidence=spatial_config.label_confidence,
                    method="heuristic_pattern",
                    features={"pattern": pattern},
                )

    # Abstract keyword
    if text.lower().startswith("abstract"):
        return ZoneClassification(
            zone=ZoneType.ABSTRACT,
            confidence=spatial_config.position_confidence,
            method="heuristic_pattern",
            features={"keyword": "abstract"},
        )

    return None


def classify_by_geometry(element: Dict, page_dims: Dict[str, int]) -> Optional[ZoneClassification]:
    """
    Classify zone based on element geometry (aspect ratio, size).

    Useful for detecting figures, tables, equations.

    Args:
        element: Layout element with bbox
        page_dims: Page dimensions

    Returns:
        ZoneClassification or None if geometry not definitive
    """
    x1 = element.get("bbox_x1", element.get("x1", 0))
    y1 = element.get("bbox_y1", element.get("y1", 0))
    x2 = element.get("bbox_x2", element.get("x2", page_dims.get("width", 0)))
    y2 = element.get("bbox_y2", element.get("y2", page_dims.get("height", 0)))

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return None

    aspect_ratio = width / height
    page_width = page_dims.get("width", 800)
    page_height = page_dims.get("height", 1000)

    # Calculate relative size
    rel_width = width / page_width if page_width > 0 else 0
    rel_height = height / page_height if page_height > 0 else 0

    # Figure: usually square-ish, medium to large size
    label = element.get("label", "").lower()
    if label == "figure":
        return ZoneClassification(
            zone=ZoneType.FIGURE,
            confidence=0.9,
            method="heuristic_label",
            features={"aspect_ratio": aspect_ratio},
        )

    # Equation: typically wide and short, centered
    if (
        0.7 < aspect_ratio
        and rel_height < 0.1
        and rel_width > 0.3
        and label in ["equation", "formula"]
    ):
        return ZoneClassification(
            zone=ZoneType.EQUATION,
            confidence=0.8,
            method="heuristic_geometry",
            features={"aspect_ratio": aspect_ratio, "rel_height": rel_height},
        )

    return None


def classify_zone_heuristic(
    element: Dict, page_dims: Dict[str, int], cross_page_stats: Optional[Dict] = None
) -> ZoneClassification:
    """
    Main heuristic classification function.
    Applies multiple heuristic rules in priority order.

    Args:
        element: Layout element with bbox, label, text
        page_dims: Page dimensions
        cross_page_stats: Optional stats about repeated elements

    Returns:
        ZoneClassification (always returns something, may be UNKNOWN)
    """
    # Priority 1: Check if marked as repeated (header/footer)
    if cross_page_stats:
        from .filters import normalize_text_for_matching

        text = element.get("text_content", element.get("text", ""))
        normalized = normalize_text_for_matching(text)

        if normalized in cross_page_stats:
            info = cross_page_stats[normalized]
            if info.zone == "header":
                return ZoneClassification(
                    zone=ZoneType.HEADER, confidence=0.95, method="heuristic_repetition"
                )
            elif info.zone == "footer":
                return ZoneClassification(
                    zone=ZoneType.FOOTER, confidence=0.95, method="heuristic_repetition"
                )

    # Priority 2: Label-based (OCR grounding labels)
    label_result = classify_by_label(element)
    if label_result and label_result.confidence >= 0.8:
        return label_result

    # Priority 3: Text pattern matching
    pattern_result = classify_by_text_pattern(element)
    if pattern_result and pattern_result.confidence >= 0.8:
        return pattern_result

    # Priority 4: Position-based
    position_result = classify_by_position(element, page_dims)
    if position_result and position_result.confidence >= 0.7:
        return position_result

    # Priority 5: Geometry-based
    geometry_result = classify_by_geometry(element, page_dims)
    if geometry_result and geometry_result.confidence >= 0.7:
        return geometry_result

    # Priority 6: Lower confidence results
    if label_result:
        return label_result
    if pattern_result:
        return pattern_result
    if position_result:
        return position_result
    if geometry_result:
        return geometry_result

    # Fallback: classify as main_text (most common)
    return ZoneClassification(zone=ZoneType.MAIN_TEXT, confidence=0.5, method="fallback")


def classify_zones_batch(
    elements: List[Dict], page_dims: Dict[str, int], cross_page_stats: Optional[Dict] = None
) -> List[Dict]:
    """
    Classify zones for a batch of elements.

    Args:
        elements: List of layout elements
        page_dims: Page dimensions
        cross_page_stats: Optional repetition stats

    Returns:
        List of elements with 'zone' and 'zone_confidence' added
    """
    results = []

    for elem in elements:
        classification = classify_zone_heuristic(elem, page_dims, cross_page_stats)

        elem_with_zone = {
            **elem,
            "zone": classification.zone.value,
            "zone_type": classification.zone,
            "zone_confidence": classification.confidence,
            "zone_method": classification.method,
        }

        results.append(elem_with_zone)

    return results


def get_zone_priority(zone: ZoneType) -> int:
    """Get reading order priority for a zone type."""
    return ZONE_PRIORITY.get(zone, 5)
