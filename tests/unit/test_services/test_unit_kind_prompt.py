"""The §2.2 prompt calls every unit a "chapter", including the appendices.

On N4.11.160 both appendix entries described themselves as "Chương này" — the
prompt says "book chapter" five times and never says which unit this is, so the
model had no way to know it was writing about an appendix.

The renderer corrects the self-reference mechanically, but a backstop that
rewrites the model's words is worse than the model getting it right: the prompt
is where the unit kind is actually known.

The second constraint closes the other half of the same defect — the appendix C
entry opened "Phụ lục B tập trung vào…", naming a different unit. A digest entry
whose heading is already printed never needs to name its unit by number, and a
number the model chooses is a number it can get wrong.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.main_content_service import MainContentService

_LONG = "Содержание раздела о числах с плавающей точкой. " * 40


def _llm():
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="Tóm tắt.")
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    return llm


def _node(title):
    return {"title": title, "content": "", "children": [{"content": _LONG, "children": []}]}


async def _prompt(title):
    llm = _llm()
    await MainContentService()._summarize_chapter(llm, _node(title), 1)
    return llm.chat_completion.await_args.args[0]


class TestUnitKindIsStated:
    @pytest.mark.asyncio
    async def test_an_appendix_is_called_an_appendix(self):
        prompt = await _prompt("Приложение Б. Числа с плавающей точкой")

        assert "appendix" in prompt.lower()
        assert "phụ lục này" in prompt.lower()

    @pytest.mark.asyncio
    async def test_a_chapter_is_still_called_a_chapter(self):
        prompt = await _prompt("Глава 4. Уровень микроархитектуры")

        assert "chương này" in prompt.lower()
        assert "appendix" not in prompt.lower()

    @pytest.mark.asyncio
    async def test_an_untitled_unit_does_not_crash(self):
        prompt = await _prompt("Số dấu phẩy động")

        assert "chương này" in prompt.lower()


class TestNamingByNumberIsForbidden:
    @pytest.mark.asyncio
    async def test_the_prompt_forbids_naming_the_unit_by_number(self):
        prompt = await _prompt("Приложение В. Программирование на языке ассемблера")

        assert "number" in prompt.lower()
        assert "Phụ lục B" in prompt, "the observed mislabel should be the worked example"
