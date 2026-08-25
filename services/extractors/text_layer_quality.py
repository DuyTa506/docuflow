"""PDF text-layer quality gate (character n-gram fluency).

`classify_pages` used to accept any page with >= N extracted characters. That
mislabels PDFs whose text layer is long but unreadable (broken ToUnicode /
custom-font encoding) — e.g. Vietnamese công văn that extract as
"THANH TRA CHNH PHU ..." with zero diacritics. Those pages must go through
DeepSeek OCR instead of Docling text extraction.

Pipeline (when quality_gate is on):
  1. length < min_chars → scanned
  2. structural corruption (NUL / U+FFFD / Private Use) → scanned
  3. too few letters/CJK to judge → text (numeric tables; OCR would not help)
  4. Latin-heavy, ~0 Vietnamese diacritics, but folded VI function words → scanned
  5. Bucket chars by Unicode script (latin / cyrillic / cjk). Score each large
     bucket with its LM(s). Latin is further split into sentences so VI+EN
     bilingual pages are not scored as one mixed string. Page passes iff every
     large bucket is fluent.
  6. else → scanned
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from services.extractors.text_layer_lm.char_lm import CharNgramLM, load_model, load_thresholds

logger = logging.getLogger(__name__)

LANGS = ("en", "zh", "ru", "vi")

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_CYRILLIC_RE = re.compile(r"[Ѐ-ԯ]")
_VI_DIACRITIC_RE = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯ\u1EA0-\u1EF9]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]")

# Fold Vietnamese diacritics for the unaccented-VI trap.
_VI_FOLD = str.maketrans(
    {
        "à": "a",
        "á": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ă": "a",
        "ằ": "a",
        "ắ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "â": "a",
        "ầ": "a",
        "ấ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "è": "e",
        "é": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ề": "e",
        "ế": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "ì": "i",
        "í": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ò": "o",
        "ó": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ồ": "o",
        "ố": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ờ": "o",
        "ớ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ù": "u",
        "ú": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ừ": "u",
        "ứ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ỳ": "y",
        "ý": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
        "đ": "d",
    }
)

# Accented forms are folded before matching; these are closed-class markers that
# dominate real Vietnamese prose and survive (badly) in unaccented encoding trash.
_VI_FUNCTION_FOLDED = frozenset(
    """
    va cua cho tu den trong ngoai tren duoi giua voi cac nhung mot la co khong
    duoc bi phai can se da dang cung chi rat nay do kia theo bang qua sau truoc
    ve tai toi nen vi neu khi luc hay nhung ma con thi quan ly su dung dat dai
    quy hoach xay dung thanh tra chinh phu uy ban nhan dan nghi quyet thong bao
    ket luan kien nghi tinh huyen thanh pho
    """.split()
)

_PUA_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？\n])\s+|\n+")

# Minimum letters+CJK before we trust perplexity / VI trap.
_MIN_SCRIPT_CHARS = 40
# A script bucket smaller than this is ignored (page numbers, short glosses).
_MIN_BUCKET_CHARS = 40
# Latin bilingual: fraction of sentence chars that must be fluent.
_LATIN_FLUENT_FRAC = 0.55
# Diacritic share of Latin letters below this → "unaccented" for VI trap.
_VI_DIACRITIC_MAX_RATIO = 0.02
# Folded VI function-word token share of alphabetic tokens.
_VI_FUNC_MIN_RATIO = 0.08
# Structural corruption: fraction of chars that are NUL/FFFD/PUA.
_CORRUPT_RATIO = 0.02
_CORRUPT_ABS = 8

# script → LMs that can score it
_BUCKET_LANGS = {
    "latin": ("en", "vi"),
    "cyrillic": ("ru",),
    "cjk": ("zh",),
}


@dataclass
class TextLayerQuality:
    usable: bool
    reason: str
    best_lang: Optional[str] = None
    perplexities: dict[str, float] = field(default_factory=dict)
    diacritic_ratio: Optional[float] = None


@lru_cache(maxsize=1)
def _models() -> dict[str, CharNgramLM]:
    return {lang: load_model(lang) for lang in LANGS}


@lru_cache(maxsize=1)
def _thresholds() -> dict:
    return load_thresholds()


def _structural_corrupt(text: str) -> bool:
    if not text:
        return False
    bad = text.count("\x00") + text.count("\ufffd") + len(_PUA_RE.findall(text))
    if bad >= _CORRUPT_ABS:
        return True
    return (bad / max(1, len(text))) >= _CORRUPT_RATIO


def _script_counts(text: str) -> tuple[int, int, int]:
    letters = len(_LETTER_RE.findall(text))
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_LETTER_RE.findall(text))
    return letters, cjk, latin


def _char_script(ch: str) -> Optional[str]:
    """Map one character to latin / cyrillic / cjk, or None (digits/punct/space)."""
    if _CJK_RE.match(ch):
        return "cjk"
    if _CYRILLIC_RE.match(ch):
        return "cyrillic"
    if _LATIN_LETTER_RE.match(ch):
        return "latin"
    return None


def script_buckets(text: str) -> dict[str, str]:
    """Pool letters by Unicode script. Not NLP sentence segmentation.

    Walk each character; Latin/Cyrillic/CJK go into separate bags. Spaces and
    sentence punctuation stay with the current bag so bilingual Latin can still
    be split on `.` / `!` / `?` later. Digits and other marks are skipped.
    """
    bags: dict[str, list[str]] = {"latin": [], "cyrillic": [], "cjk": []}
    last: Optional[str] = None
    for ch in text:
        kind = _char_script(ch)
        if kind is not None:
            bags[kind].append(ch)
            last = kind
        elif last is not None and (ch.isspace() or ch in ".!?。！？;；:\n"):
            bags[last].append(ch if ch != "\n" else " ")
        # else: digit / other punct — ignore
    return {k: "".join(v) for k, v in bags.items() if v}


def eligible_langs(text: str) -> list[str]:
    """LMs whose script appears enough on the page (for diagnostics / calibration)."""
    buckets = script_buckets(text)
    out: list[str] = []
    for script, langs in _BUCKET_LANGS.items():
        body = buckets.get(script, "")
        if sum(1 for c in body if not c.isspace()) >= _MIN_BUCKET_CHARS:
            out.extend(langs)
    return out


def _fold_vi(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower().translate(_VI_FOLD)


def _vietnamese_unaccented_trap(text: str) -> tuple[bool, Optional[float]]:
    """True when the page looks like broken Vietnamese (no tones, VI skeleton)."""
    letters, cjk, latin = _script_counts(text)
    if cjk > letters * 0.3:
        return False, None
    if latin < _MIN_SCRIPT_CHARS:
        return False, None
    # If Cyrillic or CJK dominate the page, don't apply the Latin VI trap.
    cyr = len(_CYRILLIC_RE.findall(text))
    if cjk >= _MIN_BUCKET_CHARS or cyr >= _MIN_BUCKET_CHARS:
        # Only trap when Latin is still a major share.
        if latin < max(cjk, cyr):
            return False, None
    diac = len(_VI_DIACRITIC_RE.findall(text))
    ratio = diac / max(1, latin)
    if ratio > _VI_DIACRITIC_MAX_RATIO:
        return False, ratio
    folded = _fold_vi(text)
    tokens = re.findall(r"[a-z]{2,}", folded)
    if len(tokens) < 20:
        return False, ratio
    hits = sum(1 for t in tokens if t in _VI_FUNCTION_FOLDED)
    if hits / len(tokens) >= _VI_FUNC_MIN_RATIO:
        return True, ratio
    # Also catch consonant-salad headers like "CHNH PHU HQI" with few vowels.
    sample = re.sub(r"[^A-Za-z]", "", text[:2000])
    if len(sample) >= 80:
        vowels = sum(1 for c in sample.lower() if c in "aeiouy")
        if vowels / len(sample) < 0.22 and hits >= 4:
            return True, ratio
    return False, ratio


def _lang_threshold(lang: str, thresholds: dict, global_t: float) -> float:
    if lang in thresholds:
        return float(thresholds[lang])
    return global_t


def _score_langs(
    text: str,
    langs: tuple[str, ...],
    models: dict[str, CharNgramLM],
    thresholds: dict,
    global_t: float,
) -> tuple[bool, Optional[str], dict[str, float]]:
    """True if any listed LM scores under its ceiling on `text`."""
    perplexities: dict[str, float] = {}
    best_lang = None
    best_ppl = float("inf")
    for lang in langs:
        ppl = models[lang].perplexity(text)
        perplexities[lang] = ppl
        if ppl < best_ppl:
            best_ppl = ppl
            best_lang = lang
        if ppl < _lang_threshold(lang, thresholds, global_t):
            return True, lang, perplexities
    return False, best_lang, perplexities


def _latin_sentences(text: str) -> list[str]:
    """Split Latin bag into sentence-ish units (punctuation / newlines)."""
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    if not parts:
        return [text.strip()] if text.strip() else []
    # Merge crumbs into neighbors so tiny "OK." fragments don't decide alone.
    merged: list[str] = []
    buf = ""
    for p in parts:
        cand = f"{buf} {p}".strip() if buf else p
        letters = sum(1 for c in cand if c.isalpha())
        if letters < 30:
            buf = cand
            continue
        merged.append(cand)
        buf = ""
    if buf:
        if merged:
            merged[-1] = f"{merged[-1]} {buf}".strip()
        else:
            merged.append(buf)
    return merged


def _score_latin_bucket(
    text: str,
    models: dict[str, CharNgramLM],
    thresholds: dict,
    global_t: float,
) -> tuple[bool, Optional[str], dict[str, float]]:
    """Score Latin bilingual text sentence-by-sentence (en/vi).

    A mixed VI+EN page fails whole-string PPL for both models; splitting lets
    each sentence pick the better of en/vi. Bucket passes when enough letter
    mass sits in fluent sentences.
    """
    sents = _latin_sentences(text)
    if len(sents) <= 1:
        return _score_langs(text, _BUCKET_LANGS["latin"], models, thresholds, global_t)

    total_letters = 0
    fluent_letters = 0
    all_ppl: dict[str, float] = {}
    best_lang = None
    best_ppl = float("inf")
    for sent in sents:
        letters = sum(1 for c in sent if c.isalpha())
        if letters < 20:
            continue
        total_letters += letters
        ok, lang, ppl = _score_langs(sent, _BUCKET_LANGS["latin"], models, thresholds, global_t)
        for k, v in ppl.items():
            # Keep the best (lowest) per-lang sighting for diagnostics.
            if k not in all_ppl or v < all_ppl[k]:
                all_ppl[k] = v
        if lang and ppl.get(lang, float("inf")) < best_ppl:
            best_ppl = ppl[lang]
            best_lang = lang
        if ok:
            fluent_letters += letters

    if total_letters < _MIN_BUCKET_CHARS:
        return _score_langs(text, _BUCKET_LANGS["latin"], models, thresholds, global_t)

    return (
        (fluent_letters / total_letters) >= _LATIN_FLUENT_FRAC,
        best_lang,
        all_ppl,
    )


def evaluate_text_layer(text: str, *, quality_gate: bool = True) -> TextLayerQuality:
    """Score whether extracted page text is usable without OCR."""
    body = (text or "").replace("\x00", "")  # strip NUL for scoring; still count below
    raw = text or ""
    if not quality_gate:
        return TextLayerQuality(usable=True, reason="gate_off")

    if _structural_corrupt(raw):
        return TextLayerQuality(usable=False, reason="structural_corrupt")

    letters, cjk, _latin = _script_counts(body)
    script = letters + cjk
    if script < _MIN_SCRIPT_CHARS:
        return TextLayerQuality(usable=True, reason="insufficient_script")

    trapped, dia_ratio = _vietnamese_unaccented_trap(body)
    if trapped:
        return TextLayerQuality(
            usable=False,
            reason="vietnamese_unaccented",
            diacritic_ratio=dia_ratio,
        )

    try:
        models = _models()
        thr = _thresholds()
    except FileNotFoundError as exc:
        logger.warning("text-layer LM artifacts missing (%s); accepting text page", exc)
        return TextLayerQuality(usable=True, reason="models_missing")

    thresholds = thr.get("thresholds") or {}
    global_t = float(thr.get("global_threshold") or 50.0)
    buckets = script_buckets(body)
    large = {
        name: blob
        for name, blob in buckets.items()
        if sum(1 for c in blob if not c.isspace()) >= _MIN_BUCKET_CHARS
    }
    if not large:
        return TextLayerQuality(
            usable=True, reason="insufficient_script", diacritic_ratio=dia_ratio
        )

    perplexities: dict[str, float] = {}
    best_lang = None
    best_ppl = float("inf")
    failed_bucket = None

    for script_name, blob in large.items():
        langs = _BUCKET_LANGS[script_name]
        if script_name == "latin":
            ok, lang, ppl = _score_latin_bucket(blob, models, thresholds, global_t)
        else:
            ok, lang, ppl = _score_langs(blob, langs, models, thresholds, global_t)
        for k, v in ppl.items():
            if k not in perplexities or v < perplexities[k]:
                perplexities[k] = v
        if lang and ppl.get(lang, float("inf")) < best_ppl:
            best_ppl = ppl[lang]
            best_lang = lang
        if not ok:
            failed_bucket = script_name
            break

    if failed_bucket is None:
        return TextLayerQuality(
            usable=True,
            reason="fluent",
            best_lang=best_lang,
            perplexities=perplexities,
            diacritic_ratio=dia_ratio,
        )

    return TextLayerQuality(
        usable=False,
        reason="high_perplexity",
        best_lang=best_lang,
        perplexities=perplexities,
        diacritic_ratio=dia_ratio,
    )


def classify_extracted_text(
    text: str,
    *,
    min_chars: int = 50,
    quality_gate: bool = True,
) -> str:
    """Return 'text' or 'scanned' for one page's extracted string."""
    body = (text or "").strip()
    if len(body) < min_chars:
        return "scanned"
    if not quality_gate:
        return "text"
    q = evaluate_text_layer(body, quality_gate=True)
    return "text" if q.usable else "scanned"


def reset_model_cache() -> None:
    """Test helper — drop lazy-loaded models/thresholds."""
    _models.cache_clear()
    _thresholds.cache_clear()
