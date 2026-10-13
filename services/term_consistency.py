"""One rewrite pass that makes a digest text use the termbase's Vietnamese for its terms.

The termbase is already in every generating prompt, and the model still wrote «nội đạn
học» beside «Đạn đạo trong nòng» about half the time. A regex cannot fix that safely
(the deviating phrase has no fixed length), so a separate, narrow call does — and its
answer is only accepted when nothing but wording changed.
"""

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

# The rewrite may differ in length by this much: a term swap changes a few words.
_LENGTH_RATIO = (0.85, 1.15)


def safe_revision(original: str, revised: str) -> bool:
    """True when ``revised`` can stand in for ``original``: same numbers, similar length."""
    if not revised.strip() or not original.strip():
        return False
    low, high = _LENGTH_RATIO
    if not low <= len(revised) / len(original) <= high:
        return False
    return re.findall(r"\d+(?:[.,]\d+)?", original) == re.findall(r"\d+(?:[.,]\d+)?", revised)


async def enforce_termbase(llm, text: str, glossary: Dict[str, str]) -> str:
    """``text`` with every occurrence of a glossary term worded as the glossary says.

    Never fatal: any failure, or an answer that changes more than wording, keeps the
    original text.
    """
    folded = (text or "").casefold()
    hits = [(src, vi) for src, vi in glossary.items() if src.casefold() in folded]
    if not hits:
        return text

    terms = "\n".join(f"- {src} → {vi}" for src, vi in hits)
    prompt = (
        "You are a terminology editor for a library catalogue.\n\n"
        "TASK: Rewrite the text below so that each term in the list is written with "
        "exactly the Vietnamese given for it, everywhere it appears.\n\n"
        "CONSTRAINTS:\n"
        "- The text may name a term in another Vietnamese wording or leave it in the "
        "source language — replace that wording with the one in the list.\n"
        "- Keep the source-language term in parentheses where the text already has it.\n"
        "- Change NOTHING else: same sentences, same order, same numbers, same names.\n"
        "- Do not add, remove or reorder information.\n\n"
        f"TERMS (source → required Vietnamese):\n{terms}\n\n"
        f"TEXT:\n{text}\n\n"
        "OUTPUT: the rewritten text only, no comments.\n"
    )
    try:
        revised = (await llm.chat_completion(prompt, max_tokens=max(512, len(text)))).strip()
    except Exception as exc:
        logger.warning("Term consistency pass failed (%s) — keeping the text", exc)
        return text
    if not safe_revision(text, revised):
        logger.warning("Term consistency pass changed more than wording — keeping the text")
        return text
    return revised
