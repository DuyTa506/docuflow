"""Translate merged layout blocks; preserve bbox at block level."""

from __future__ import annotations

from typing import Awaitable, Callable, List, Optional

from config.settings import settings
from core.pageindex.enrichment.translator import StructuredTranslator
from services.translators._parallel import ProgressCallback, run_parallel
from utils.markdown_docx import _IMAGE_LABELS as _CAPTIONED_LABELS
from utils.markdown_docx import _IMG_PLACEHOLDER_RE
from utils.table_translate import translate_table_text
from utils.translation_blocks import (
    TranslationBlock,
    block_to_translated_element,
    merge_payloads_to_blocks,
)
from utils.translation_elements import flatten_translated_elements


class BlockTranslator:
    """Merge elements into blocks, translate in parallel, emit spatial payloads."""

    def __init__(self, translator: StructuredTranslator):
        self.translator = translator

    async def translate_payloads(
        self,
        payloads: List[dict],
        *,
        on_progress: ProgressCallback = None,
    ) -> dict:
        blocks = merge_payloads_to_blocks(payloads)
        translated = await run_parallel(
            blocks,
            self._translate_block,
            parallelism=settings.translation_parallelism,
            on_progress=on_progress,
            progress_label="Khối",
        )
        flat: List[dict] = []
        for item in translated:
            flat.extend(item)

        return {
            "translation_mode": "block_based",
            "translated_elements": flat,
            "translated_content": flatten_translated_elements(flat),
            "translated_file_path": None,
        }

    async def _translate_block(self, _idx: int, block: TranslationBlock) -> List[dict]:
        if block.passthrough:
            return [await self._translate_passthrough(dict(p)) for p in block.source_payloads]

        source_text = (block.text or "").strip()
        if not source_text:
            return [
                block_to_translated_element(block, "") for _ in block.source_payloads or [None]
            ] or [block_to_translated_element(block, "")]

        if block.is_heading:
            translated = await self.translator.translate_title(source_text)
        else:
            # Long TOC / multi-paragraph blocks auto-chunk inside translate_text.
            translated = await self.translator.translate_text(source_text)

        return [block_to_translated_element(block, translated)]

    async def _translate_passthrough(self, payload: dict) -> dict:
        """Tables get their cells translated and figures their caption; the
        markup, image and bbox stay as they are."""
        label = (payload.get("label") or "").lower()
        text = (payload.get("text_content") or "").strip()
        if not text:
            return payload
        if label == "table":
            translated = await translate_table_text(text, self.translator.translate_cells)
            if translated is None:
                translated = await self.translator.translate_text(text)
            payload["text_content"] = translated
        elif label in _CAPTIONED_LABELS and not _IMG_PLACEHOLDER_RE.match(text):
            payload["text_content"] = await self.translator.translate_title(text)
        return payload

