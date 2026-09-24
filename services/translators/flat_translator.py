"""Legacy flat-text chunk translation (last-resort fallback)."""

from __future__ import annotations

from typing import Any, Callable, Optional

from config.settings import settings
from core.pageindex.enrichment.translator import StructuredTranslator
from services.translators._parallel import run_parallel

ProgressCallback = Optional[Callable[[int, str], Any]]


class FlatTranslator:
    """Translate a flat string in token-bounded chunks."""

    def __init__(self, translator: StructuredTranslator):
        self.translator = translator

    async def translate_text(
        self,
        text: str,
        *,
        on_progress: ProgressCallback = None,
    ) -> dict:
        chunks = self.translator.chunk_text(text, max_tokens=self.translator.chunk_size)

        async def _translate_chunk(_idx: int, chunk: str) -> str:
            return await self.translator.translate_text(chunk)

        translated_parts = await run_parallel(
            chunks,
            _translate_chunk,
            parallelism=settings.translation_parallelism,
            on_progress=on_progress,
            progress_label="Phần",
        )

        return {
            "translation_mode": "flat",
            "translated_elements": None,
            "translated_content": "\n\n".join(translated_parts),
            "translated_file_path": None,
        }
