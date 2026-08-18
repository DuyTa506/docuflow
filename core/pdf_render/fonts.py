"""Font resolution for the hybrid PDF renderer."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

_FONT_CACHE: dict[str, str] = {}


def resolve_vi_font_path() -> Optional[str]:
    """Prefer a font whose space glyph is U+0020 so search/copy stay usable.

    Times New Roman (and Liberation Serif) map space to U+00A0, which breaks
    PDF text search. DejaVu covers Vietnamese tones and uses a real space.
    """
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    ):
        if os.path.isfile(candidate):
            return candidate
    try:
        from config.settings import settings

        path = settings.pdf_overlay_vi_font_path
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    return None


@lru_cache(maxsize=4)
def resolve_render_font_path(lang: str = "vi") -> Optional[str]:
    code = (lang or "vi").lower().split("-")[0]
    if code == "vi":
        path = resolve_vi_font_path()
        if path:
            return path
    try:
        from babeldoc.assets.assets import get_font_and_metadata

        font_path, _ = get_font_and_metadata("GoNotoKurrent-Regular.ttf")
        return font_path.as_posix()
    except Exception:
        return resolve_vi_font_path()


def fitz_font(lang: str = "vi"):
    import fitz

    path = resolve_render_font_path(lang)
    if path:
        return fitz.Font(fontfile=path)
    return fitz.Font("helv")
