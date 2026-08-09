"""Translate document content block-by-block (layout elements)."""

from __future__ import annotations

from typing import Awaitable, Callable, List, Optional

from config.settings import settings
from core.pageindex.enrichment.translator import StructuredTranslator
from services.translators._parallel import ProgressCallback, run_parallel
from utils.translation_elements import (
    flatten_translated_elements,
    is_heading_label,
    should_skip_label,
)


class ElementTranslator:
    """Translate each layout element; preserve bbox, label, and reading order."""

    def __init__(self, translator: StructuredTranslator):
        self.translator = translator

    async def translate_payloads(
        self,
        payloads: List[dict],
        *,
        on_progress: ProgressCallback = None,
    ) -> dict:
        flat = await run_parallel(
            payloads,
            self._translate_one,
            parallelism=settings.translation_parallelism,
            on_progress=on_progress,
            progress_label="Element",
        )
        return {
            "translation_mode": "element_based",
            "translated_elements": flat,
            "translated_content": flatten_translated_elements(flat),
            "translated_file_path": None,
        }

    async def _translate_one(self, _idx: int, payload: dict) -> dict:
        label = payload.get("label") or "text"
        source_text = (payload.get("text_content") or "").strip()
        out = dict(payload)

        if should_skip_label(label) or not source_text:
            return out
        if is_heading_label(label):
            out["text_content"] = await self.translator.translate_title(source_text)
        else:
            out["text_content"] = await self.translator.translate_text(source_text)
        return out
