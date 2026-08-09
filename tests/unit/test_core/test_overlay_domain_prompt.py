"""PDF overlay must carry the same domain terminology guidance as the other
translation paths — it previously used a minimal generic prompt, silently
dropping the user's requested domain on the highest-fidelity output mode.
"""

from unittest.mock import MagicMock, patch

from core.pageindex.enrichment.translator import DOMAIN_INSTRUCTIONS


def _adapter(domain):
    with patch("openai.OpenAI"):
        from core.pdf_overlay.llm_adapter import OverlayLLMAdapter

        return OverlayLLMAdapter(source_lang="en", target_lang="vi", domain=domain)


def test_overlay_prompt_carries_domain_instruction():
    adapter = _adapter("military")
    fake = MagicMock()
    fake.choices = [MagicMock()]
    fake.choices[0].message.content = "bản dịch"
    adapter._sync.chat.completions.create = MagicMock(return_value=fake)

    adapter.translate("Some tactical text {v0}")

    prompt = adapter._sync.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert DOMAIN_INSTRUCTIONS["military"].split(".")[0] in prompt
    # placeholder preservation rule must survive
    assert "{v0}" in prompt and "placeholder" in prompt.lower()
    # terminology constraints from the main paths
    assert "proper nouns" in prompt.lower()


def test_pdf_overlay_translator_forwards_domain():
    import inspect

    from services.translators.pdf_overlay_translator import PdfOverlayTranslator

    sig = inspect.signature(PdfOverlayTranslator.translate_file)
    assert "domain" in sig.parameters
