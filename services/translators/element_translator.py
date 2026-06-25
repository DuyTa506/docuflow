"""Translate document content block-by-block (layout elements)."""

from __future__ import annotations

from typing import Awaitable, Callable, List, Optional

from core.pageindex.enrichment.translator import StructuredTranslator
from data.db_models import LayoutElement
from utils.translation_elements import (
    flatten_translated_elements,
    is_heading_label,
    layout_element_to_dict,
    should_skip_label,
)


ProgressCallback = Optional[Callable[[int, str], Awaitable[None] | None]]


class ElementTranslator:
    """Translate each layout element; preserve bbox, label, and reading order."""

    def __init__(self, translator: StructuredTranslator):
        self.translator = translator

    async def translate_elements(
        self,
        elements: List[LayoutElement],
        *,
        on_progress: ProgressCallback = None,
    ) -> dict:
        from utils.translation_elements import layout_element_to_dict

        payloads = [
            layout_element_to_dict(elem, elem.page.page_number if elem.page else 1)
            for elem in elements
        ]
        return await self.translate_payloads(payloads, on_progress=on_progress)

    async def translate_payloads(
        self,
        payloads: List[dict],
        *,
        on_progress: ProgressCallback = None,
    ) -> dict:
        translated: List[dict] = []
        total = len(payloads)

        for idx, payload in enumerate(payloads):
            label = payload.get("label") or "text"
            source_text = (payload.get("text_content") or "").strip()
            out = dict(payload)

            if should_skip_label(label) or not source_text:
                translated.append(out)
            elif is_heading_label(label):
                out["text_content"] = await self.translator.translate_title(source_text)
                translated.append(out)
            else:
                out["text_content"] = await self.translator.translate_text(source_text)
                translated.append(out)

            if on_progress and total:
                pct = int(((idx + 1) / total) * 95)
                await _maybe_await(on_progress(pct, f"Element {idx + 1}/{total}"))

        return {
            "translation_mode": "element_based",
            "translated_elements": translated,
            "translated_content": flatten_translated_elements(translated),
            "translated_file_path": None,
        }


async def _maybe_await(result):
    if result is not None and hasattr(result, "__await__"):
        await result
