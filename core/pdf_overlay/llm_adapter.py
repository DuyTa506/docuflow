"""Sync LLM adapter for PDF overlay paragraph translation."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from config.settings import lang_name

logger = logging.getLogger(__name__)


class OverlayLLMAdapter:
    """Thin sync wrapper around the async pipeline LLM client."""

    def __init__(
        self,
        llm_client,
        *,
        source_lang: str,
        target_lang: str,
        domain: str = "general",
    ):
        self._client = llm_client
        self._source_lang = lang_name(source_lang)
        self._target_lang = lang_name(target_lang)
        self._domain = domain
        self.lang_out = target_lang

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, self._translate_async(text)).result()
            return loop.run_until_complete(self._translate_async(text))
        except RuntimeError:
            return asyncio.run(self._translate_async(text))

    async def _translate_async(self, text: str) -> str:
        prompt = (
            f"You are a professional translator ({self._domain} domain).\n"
            f"Translate the following text from {self._source_lang} to {self._target_lang}.\n"
            "Preserve formula placeholders like {v0}, {v1} exactly — do not translate or remove them.\n"
            "Output ONLY the translation, no explanations.\n\n"
            f"{text}"
        )
        result = await self._client.chat_completion(prompt)
        return (result or "").strip()
