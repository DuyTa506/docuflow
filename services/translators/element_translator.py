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
        translated: List[dict] = []
        total = len(elements)

        for idx, elem in enumerate(elements):
            page_number = elem.page.page_number if elem.page else 1
            payload = layout_element_to_dict(elem, page_number)
            label = elem.label or "text"
            source_text = (elem.text_content or "").strip()

            if should_skip_label(label) or not source_text:
                translated.append(payload)
            elif is_heading_label(label):
                payload["text_content"] = await self.translator.translate_title(source_text)
                translated.append(payload)
            else:
                payload["text_content"] = await self.translator.translate_text(source_text)
                translated.append(payload)

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
