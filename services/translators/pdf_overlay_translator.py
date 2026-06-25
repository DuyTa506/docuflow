"""PDF overlay translator — preserves layout and formula glyphs."""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, Optional

from config.settings import settings
from core.pdf_overlay.llm_adapter import OverlayLLMAdapter
from core.pdf_overlay.pipeline import translate_pdf_bytes

ProgressCb = Optional[Callable[[int, str], Awaitable[None]]]


class PdfOverlayTranslator:
    """Translate text-layer PDFs by re-drawing translated text at original coordinates."""

    async def translate_file(
        self,
        file_path: str,
        out_path: str,
        *,
        source_lang: str,
        target_lang: str,
        on_progress: ProgressCb = None,
    ) -> dict:
        from api.dependencies import get_llm_client

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        llm_client = get_llm_client()
        adapter = OverlayLLMAdapter(
            llm_client,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        loop = asyncio.get_event_loop()

        def _sync_progress(done: int, total: int) -> None:
            if on_progress and total:
                pct = min(95, 10 + int(80 * done / total))
                asyncio.run_coroutine_threadsafe(
                    on_progress(pct, f"PDF overlay page {done}/{total}"),
                    loop,
                )

        translated = await asyncio.to_thread(
            translate_pdf_bytes,
            pdf_bytes,
            lang_in=source_lang,
            lang_out=target_lang,
            llm_adapter=adapter,
            thread=settings.pdf_overlay_threads,
            on_progress=_sync_progress,
        )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(translated)

        if on_progress:
            await on_progress(98, "Saving translated PDF")

        return {
            "translated_file_path": out_path,
            "translation_mode": "pdf_overlay",
            "translated_content": None,
            "translated_elements": None,
        }
