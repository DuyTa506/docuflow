"""OCR prompt validation."""

from config.settings import Settings
from core.constants import OCR_PROMPTS


class TestOcrPromptValidator:
    def test_default_uses_grounding_markdown(self):
        s = Settings(ocr_prompt=OCR_PROMPTS["markdown"])
        assert "<image>" in s.ocr_prompt
        assert "<|grounding|>" in s.ocr_prompt

    def test_repairs_missing_grounding_tag(self):
        s = Settings(ocr_prompt="<image>\nConvert to markdown with LaTeX $...$")
        assert s.ocr_prompt.startswith("<image>\n<|grounding|>")

    def test_repairs_missing_image_tag(self):
        s = Settings(ocr_prompt="Convert the document to markdown.")
        assert s.ocr_prompt.startswith("<image>")

    def test_empty_falls_back_to_default(self):
        s = Settings(ocr_prompt="   ")
        assert s.ocr_prompt == OCR_PROMPTS["markdown"]
