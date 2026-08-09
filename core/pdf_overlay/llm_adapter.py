"""Sync LLM adapter for PDF overlay paragraph translation (thread-pool safe)."""

from __future__ import annotations

import logging
import os

from config.settings import lang_name, settings

logger = logging.getLogger(__name__)


class OverlayLLMAdapter:
    """Sync OpenAI-compatible client for use inside ThreadPoolExecutor workers."""

    def __init__(
        self,
        llm_client=None,
        *,
        source_lang: str,
        target_lang: str,
        domain: str = "general",
    ):
        self._source_lang = lang_name(source_lang)
        self._target_lang = lang_name(target_lang)
        self._domain = domain
        self.lang_out = target_lang

        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLAMA_API_KEY") or "local"
        kwargs = {
            "api_key": api_key,
            "timeout": settings.ai_request_timeout_seconds,
        }
        if settings.ai_openai_base_url:
            kwargs["base_url"] = settings.ai_openai_base_url
        self._sync = OpenAI(**kwargs)
        self._model = settings.ai_model

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        from core.pageindex.enrichment.translator import DOMAIN_INSTRUCTIONS

        role = DOMAIN_INSTRUCTIONS.get(self._domain, DOMAIN_INSTRUCTIONS["general"])
        prompt = (
            f"{role}\n"
            f"Translate the following text from {self._source_lang} to {self._target_lang}.\n"
            "Preserve ALL proper nouns, acronyms, codes, and technical identifiers exactly as-is.\n"
            "Preserve formula placeholders like {v0}, {v1} exactly — do not translate or remove them.\n"
            "Output ONLY the translation, no explanations.\n\n"
            f"{text}"
        )
        try:
            resp = self._sync.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            out = (resp.choices[0].message.content or "").strip()
            if not out:
                # Empty completion — raise so the bounded paragraph retry
                # (translate_paragraphs) retries, then degrades to source.
                raise ValueError("empty translation output")
            return out
        except Exception as exc:
            logger.warning("Overlay LLM call failed: %s", exc)
            raise
