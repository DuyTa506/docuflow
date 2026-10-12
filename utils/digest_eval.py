"""Scoring helpers for digest quality against a reviewer's reference answers.

Pure functions over plain data, so `scripts/eval_digest_quality.py` (which reads
the database) and the unit tests share one definition of each measure.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set


def _norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


def chapter_coverage(present: Iterable[int], expected: int) -> Dict[str, object]:
    have = {int(n) for n in present if n}
    missing = [n for n in range(1, expected + 1) if n not in have]
    return {"found": expected - len(missing), "expected": expected, "missing": missing}


def keyword_overlap(displays: Sequence[str], gold: Sequence[Sequence[str]]) -> Dict[str, object]:
    """Gold keywords found, matched on the English term (the stable side of «Việt (English)»)."""
    haystack = [_norm(d) for d in displays]
    missing = [en for _, en in gold if not any(_norm(en) in h for h in haystack)]
    return {"matched": len(gold) - len(missing), "gold": len(gold), "missing": missing}


def renderings(texts: Iterable[str], english_terms: Iterable[str]) -> Dict[str, Set[str]]:
    """English term → the distinct Vietnamese renderings written before «(term)»."""
    joined = "\n".join(texts)
    found: Dict[str, Set[str]] = {}
    for term in english_terms:
        pattern = re.compile(r"([^()\n;.,:]{2,60}?)\s*\(\s*" + re.escape(term) + r"\s*\)", re.I)
        vi = {_norm(m.group(1)) for m in pattern.finditer(joined)}
        if vi:
            found[term] = vi
    return found


def inconsistent_terms(texts: Iterable[str], english_terms: Iterable[str]) -> Dict[str, List[str]]:
    """Terms rendered more than one way. Only the last words before the term are compared,
    so «Đạn đạo trong» and «phần đạn đạo trong» do not count as two."""
    out: Dict[str, List[str]] = {}
    for term, variants in renderings(texts, english_terms).items():
        width = max(1, len(term.split()))
        tails = {" ".join(v.split()[-width:]) for v in variants}
        if len(tails) > 1:
            out[term] = sorted(tails)
    return out
