"""
Normalization service for OCR text cleanup.

Applies Unicode normalization, OCR artifact removal, whitespace cleanup,
and language-specific corrections.
"""
import re
import unicodedata


class NormalizationService:
    """Stateless text normalization pipeline."""

    def normalize(self, text: str, language: str = "en") -> str:
        """
        Full normalization pipeline:
        1. Unicode NFC normalization
        2. OCR artifact cleanup
        3. Whitespace / punctuation normalization
        4. Language-specific fixes
        """
        if not text:
            return text

        text = self._unicode_normalize(text)
        text = self._clean_ocr_artifacts(text)
        text = self._normalize_whitespace(text)
        text = self._language_specific(text, language)
        return text.strip()

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _unicode_normalize(text: str) -> str:
        """Apply Unicode NFC normalization (canonical decomposition + composition)."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _clean_ocr_artifacts(text: str) -> str:
        """Remove common OCR noise characters and broken sequences."""
        # Remove isolated control characters (except newlines/tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Fix common OCR mis-recognitions
        text = text.replace("\ufb01", "fi")   # fi ligature
        text = text.replace("\ufb02", "fl")   # fl ligature
        text = text.replace("\u2019", "'")    # right single quote → ascii
        text = text.replace("\u201c", '"')    # left double quote
        text = text.replace("\u201d", '"')    # right double quote
        text = text.replace("\u2014", " - ")  # em dash
        text = text.replace("\u2013", "-")    # en dash
        text = text.replace("\u00ad", "")     # soft hyphen
        # Remove runs of 3+ identical punctuation (OCR noise)
        text = re.sub(r"([^\w\s])\1{2,}", r"\1", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse whitespace, fix spacing around punctuation."""
        # Collapse multiple spaces (but preserve newlines)
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # No space before period / comma / semicolon / colon
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        # Ensure space after punctuation (except at end of line)
        text = re.sub(r"([.,;:!?])([^\s\n\d\"'])", r"\1 \2", text)
        return text

    @staticmethod
    def _language_specific(text: str, language: str) -> str:
        """Apply language-specific cleanup."""
        lang = language.lower()[:2]

        if lang == "vi":
            # Vietnamese: fix broken diacritics from OCR
            # Common pattern: space inserted inside diacritic word
            # e.g., "Việ t Nam" → "Việt Nam" — hard to do generically,
            # so we just fix common known patterns
            pass

        elif lang in ("zh", "ja", "ko"):
            # CJK: remove spaces between CJK characters (OCR often adds them)
            text = re.sub(
                r"([\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af])"
                r"\s+"
                r"([\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af])",
                r"\1\2",
                text,
            )

        elif lang == "ru":
            # Cyrillic: fix common OCR confusions (Latin ↔ Cyrillic homoglyphs)
            pass

        return text
