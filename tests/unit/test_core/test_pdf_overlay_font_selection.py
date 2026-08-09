"""Regression tests for pdf_overlay's target-font selection.

Confirmed live: Times New Roman has full coverage of all Vietnamese
precomposed tone-marked characters and is ~322KB, vs. GoNotoKurrent's
~15MB unsubsetted -- matches the Times New Roman convention already used
for DOCX exports. GoNotoKurrent must remain the fallback for every other
supported script (zh/ja/ko/ar/hi/th/etc.), which Times New Roman has no
glyphs for at all.
"""

from unittest.mock import patch

from core.pdf_overlay.pipeline import _download_target_font


class TestDownloadTargetFont:
    def test_vietnamese_uses_times_new_roman_when_available(self):
        with (
            patch("os.path.isfile", return_value=True),
            patch("config.settings.settings.pdf_overlay_vi_font_path", "/fake/times.ttf"),
        ):
            result = _download_target_font("vi")
        assert result == "/fake/times.ttf"

    def test_vietnamese_falls_back_to_gonotokurrent_when_font_missing(self):
        with (
            patch("os.path.isfile", return_value=False),
            patch("babeldoc.assets.assets.get_font_and_metadata") as mock_get_font,
        ):
            mock_get_font.return_value = (
                __import__("pathlib").Path("/fake/GoNotoKurrent-Regular.ttf"),
                {},
            )
            result = _download_target_font("vi")
        assert "GoNotoKurrent" in result

    def test_non_vietnamese_language_uses_gonotokurrent(self):
        with patch("babeldoc.assets.assets.get_font_and_metadata") as mock_get_font:
            mock_get_font.return_value = (
                __import__("pathlib").Path("/fake/GoNotoKurrent-Regular.ttf"),
                {},
            )
            result = _download_target_font("zh")
        assert "GoNotoKurrent" in result

    def test_language_code_with_region_suffix_still_matches_vietnamese(self):
        with (
            patch("os.path.isfile", return_value=True),
            patch("config.settings.settings.pdf_overlay_vi_font_path", "/fake/times.ttf"),
        ):
            result = _download_target_font("vi-VN")
        assert result == "/fake/times.ttf"
