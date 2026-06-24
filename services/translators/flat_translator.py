"""Legacy flat-text chunk translation (last-resort fallback)."""

from __future__ import annotations

from typing import Any, Callable, Optional

from core.pageindex.enrichment.translator import StructuredTranslator


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
        translated_parts = []
        for i, chunk in enumerate(chunks):
            translated_parts.append(await self.translator.translate_text(chunk))
            if on_progress and chunks:
                pct = int(((i + 1) / len(chunks)) * 95)
                result = on_progress(pct, f"Chunk {i + 1}/{len(chunks)}")
                if result is not None and hasattr(result, "__await__"):
                    await result

        return {
            "translation_mode": "flat",
            "translated_elements": None,
            "translated_content": "\n\n".join(translated_parts),
            "translated_file_path": None,
        }
