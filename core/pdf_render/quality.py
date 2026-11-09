"""Post-render quality gates for layout PDFs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from core.pdf_render.geometry import RENDERER_VERSION, Rect, Region
from core.pdf_render.text_layout import FittedText

_WORD_RE = re.compile(r"[A-Za-z]{4,}")


@dataclass
class QualityIssue:
    page: int
    kind: str
    message: str
    critical: bool = False


@dataclass
class PdfRenderQuality:
    ok: bool
    renderer_version: str = RENDERER_VERSION
    pdf_mode: str = "layout"
    issues: list[QualityIssue] = field(default_factory=list)
    pages_checked: int = 0
    fallback: Optional[str] = None
    missing_glyphs: int = 0
    overflow_regions: int = 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "renderer_version": self.renderer_version,
            "pdf_mode": self.pdf_mode,
            "pages_checked": self.pages_checked,
            "fallback": self.fallback,
            "missing_glyphs": self.missing_glyphs,
            "overflow_regions": self.overflow_regions,
            "issues": [
                {
                    "page": i.page,
                    "kind": i.kind,
                    "message": i.message,
                    "critical": i.critical,
                }
                for i in self.issues
            ],
        }


def _ngrams(text: str, n: int = 4) -> set[str]:
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def evaluate_page_layout(
    *,
    page_number: int,
    drawn: list[tuple[Region, Rect, FittedText]],
    source_text: str,
    output_text: str,
    font_floor_hits: int,
    figure_delta: Optional[float] = None,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for i, (region, rect, fitted) in enumerate(drawn):
        if fitted.overflow:
            issues.append(
                QualityIssue(
                    page_number,
                    "overflow",
                    f"region {region.id} overflowed {len(fitted.overflow)} chars",
                    critical=False,
                )
            )
        if fitted.missing_glyphs:
            issues.append(
                QualityIssue(
                    page_number,
                    "missing_glyph",
                    f"region {region.id} missing {fitted.missing_glyphs} glyphs",
                    critical=fitted.missing_glyphs > 8,
                )
            )
        if rect.is_empty():
            issues.append(
                QualityIssue(page_number, "bbox", f"region {region.id} has empty bbox", True)
            )
        for other_region, other_rect, _ in drawn[i + 1 :]:
            if region.column_index != other_region.column_index and not (
                region.full_width or other_region.full_width
            ):
                if (
                    rect.x_overlap_ratio(other_rect) > 0.25
                    and rect.y_overlap_ratio(other_rect) > 0.25
                ):
                    issues.append(
                        QualityIssue(
                            page_number,
                            "column_overlap",
                            f"{region.id} overlaps {other_region.id}",
                            critical=True,
                        )
                    )
            elif (
                rect.intersects(other_rect, eps=1.0)
                and rect.intersection_over_self(other_rect) > 0.35
            ):
                issues.append(
                    QualityIssue(
                        page_number,
                        "block_overlap",
                        f"{region.id} overlaps {other_region.id}",
                        critical=True,
                    )
                )
    if font_floor_hits > max(3, len(drawn) // 2):
        issues.append(
            QualityIssue(
                page_number,
                "font_floor",
                f"{font_floor_hits} regions at minimum font size",
                critical=False,
            )
        )
    src_grams = _ngrams((source_text or "").replace("\xa0", " "))
    out_lower = (output_text or "").replace("\xa0", " ").lower()
    leftover = [g for g in list(src_grams)[:80] if g in out_lower]
    if leftover and len(leftover) >= 10:
        issues.append(
            QualityIssue(
                page_number,
                "source_ngram",
                f"{len(leftover)} source n-grams still present",
                critical=len(leftover) >= 18,
            )
        )
    drawn_chars = sum(len((t.visible_text or "").replace("\xa0", " ")) for _, _, t in drawn)
    actual = len((output_text or "").replace("\xa0", " ").strip())
    if drawn_chars > 40 and actual < max(20, int(0.35 * drawn_chars)):
        issues.append(
            QualityIssue(
                page_number,
                "text_layer",
                "output text much shorter than fitted regions",
                critical=True,
            )
        )
    if figure_delta is not None and figure_delta > 0.08:
        issues.append(
            QualityIssue(
                page_number,
                "figure_delta",
                f"figure/table pixels changed by {figure_delta:.0%}",
                critical=True,
            )
        )
    return issues


def aggregate_quality(
    issues: list[QualityIssue], pages_checked: int, pdf_mode: str
) -> PdfRenderQuality:
    critical = [i for i in issues if i.critical]
    missing = sum(1 for i in issues if i.kind == "missing_glyph")
    overflow = sum(1 for i in issues if i.kind == "overflow")
    ok = not critical
    fallback = None if ok else "reflow"
    return PdfRenderQuality(
        ok=ok,
        pdf_mode=pdf_mode,
        issues=issues,
        pages_checked=pages_checked,
        fallback=fallback,
        missing_glyphs=missing,
        overflow_regions=overflow,
    )
