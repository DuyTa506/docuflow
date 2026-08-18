"""Word-wrap and font-size fitting measured with real font metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.pdf_render.geometry import Rect

MIN_FONT_PT = 5.0
MAX_FONT_PT = 12.0
LINE_HEIGHT_RATIO = 1.28


@dataclass
class FittedText:
    fontsize: float
    lines: list[str]
    overflow: str = ""
    line_height: float = 0.0
    used_height: float = 0.0
    missing_glyphs: int = 0

    @property
    def visible_text(self) -> str:
        return "\n".join(self.lines)


def _tokenize(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    tokens: list[str] = []
    for raw_line in text.split("\n"):
        if not raw_line:
            tokens.append("\n")
            continue
        parts = raw_line.split(" ")
        for i, part in enumerate(parts):
            if i:
                tokens.append(" ")
            if part:
                tokens.append(part)
        tokens.append("\n")
    if tokens and tokens[-1] == "\n":
        tokens.pop()
    return tokens


def wrap_words(text: str, font, fontsize: float, max_width: float) -> list[str]:
    if max_width <= 1:
        return [text] if text else []
    lines: list[str] = [""]
    for token in _tokenize(text):
        if token == "\n":
            lines.append("")
            continue
        candidate = lines[-1] + token
        width = font.text_length(candidate, fontsize=fontsize) if candidate else 0.0
        if width <= max_width or not lines[-1]:
            lines[-1] = candidate.lstrip(" ") if not lines[-1] else candidate
            continue
        if token == " ":
            lines.append("")
            continue
        # Hard-split an overlong token.
        if font.text_length(token, fontsize=fontsize) > max_width:
            chunk = token
            if lines[-1]:
                lines.append("")
            while chunk:
                lo, hi = 1, len(chunk)
                fit = 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if font.text_length(chunk[:mid], fontsize=fontsize) <= max_width:
                        fit = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                lines[-1] = chunk[:fit]
                chunk = chunk[fit:]
                if chunk:
                    lines.append("")
        else:
            lines.append(token.lstrip(" "))
    return [ln.rstrip() for ln in lines if ln is not None]


def count_missing_glyphs(text: str, font) -> int:
    missing = 0
    has_glyph = getattr(font, "has_glyph", None)
    if not callable(has_glyph):
        return 0
    for ch in text:
        if ch.isspace():
            continue
        try:
            if not has_glyph(ord(ch)):
                missing += 1
        except Exception:
            missing += 1
    return missing


def fit_textbox(
    text: str,
    rect: Rect,
    font,
    *,
    min_pt: float = MIN_FONT_PT,
    max_pt: float = MAX_FONT_PT,
    line_height_ratio: float = LINE_HEIGHT_RATIO,
    bold: bool = False,
) -> FittedText:
    text = (text or "").strip()
    if not text or rect.width <= 0 or rect.height <= 0:
        return FittedText(fontsize=min_pt, lines=[], overflow=text)

    cap = min(max_pt, max_pt if not bold else max_pt)
    lo, hi = min_pt, cap
    best: Optional[FittedText] = None
    for _ in range(10):
        mid = (lo + hi) / 2.0
        lines = wrap_words(text, font, mid, max(rect.width - 2.0, 1.0))
        lh = mid * line_height_ratio
        used = len(lines) * lh
        if used <= rect.height - 0.75:
            best = FittedText(
                fontsize=mid,
                lines=lines,
                overflow="",
                line_height=lh,
                used_height=used,
                missing_glyphs=count_missing_glyphs(text, font),
            )
            lo = mid
        else:
            hi = mid
    if best is not None:
        return best

    lines = wrap_words(text, font, min_pt, max(rect.width - 1.0, 1.0))
    lh = min_pt * line_height_ratio
    max_lines = max(1, int(rect.height / lh))
    visible = lines[:max_lines]
    overflow_lines = lines[max_lines:]
    overflow = " ".join(overflow_lines).strip()
    return FittedText(
        fontsize=min_pt,
        lines=visible,
        overflow=overflow,
        line_height=lh,
        used_height=len(visible) * lh,
        missing_glyphs=count_missing_glyphs(text, font),
    )


def expand_rect_in_column(
    rect: Rect,
    *,
    needed_height: float,
    page_h: float,
    neighbors: list[Rect],
    gap: float = 4.0,
    max_ratio: float = 0.8,
) -> Rect:
    """Grow downward inside the same column, stopping at the next neighbor."""
    orig_h = rect.height
    max_y1 = min(rect.y0 + orig_h * (1.0 + max_ratio), page_h)
    for other in neighbors:
        if other.y0 <= rect.y0 + 0.5:
            continue
        if rect.x_overlap_ratio(other) < 0.5:
            continue
        max_y1 = min(max_y1, other.y0 - gap)
    target = min(max_y1, rect.y0 + needed_height)
    footer_y = page_h * 0.88
    if rect.y1 < footer_y:
        target = min(target, footer_y)
    if target > rect.y1:
        return Rect(rect.x0, rect.y0, rect.x1, target)
    return rect
