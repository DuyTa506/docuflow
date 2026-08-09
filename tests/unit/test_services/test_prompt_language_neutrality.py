"""The corpus is not English and Russian.

Two prompts named exactly those two languages and no others:

    main_content:   "add English/Russian originals in parentheses when helpful"
    bibliographic:  "Vietnamese translation in parentheses if source is English/Russian"

The library holds English, Russian, Chinese, Japanese and Vietnamese documents.
On a Chinese or Japanese source the rule names a language that is not there, so
the model is told nothing about what it is actually reading; on a Vietnamese
source the rule asks for a "translation" of text already in Vietnamese.

The keyword prompt already had the right shape — "Vietnamese term (Original
term) for non-Vietnamese docs" — which is correct for every language at once and
does not depend on `Document.source_language`, a field `utils/lang_detect`
documents as unreliable (a book with Vietnamese front matter and a Russian body
detects as "vi").
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.bibliographic_service import BibliographicService
from services.main_content_service import MainContentService

NAMED_LANGUAGES = ("English/Russian", "English or Russian")

_LONG = "内容涉及计算机体系结构与流水线技术。" * 60


def _llm(payload="Tóm tắt."):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=payload)
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.extract_json = MagicMock(return_value={})
    llm.encoding = None
    return llm


async def _main_content_prompt():
    llm = _llm()
    node = {"title": "第 3 章 数字逻辑层", "content": "", "children": [{"content": _LONG}]}
    await MainContentService()._summarize_chapter(llm, node, 3)
    return llm.chat_completion.await_args.args[0]


async def _bibliographic_prompt():
    llm = _llm()
    svc = BibliographicService()
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    with (
        patch("services.bibliographic_service.get_db_manager") as dbm,
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch.object(svc, "_read_front_matter", return_value=_LONG),
        patch.object(svc, "_progress"),
    ):
        dbm.return_value.session.return_value = session
        await svc._extract("DOC_001")
    return llm.chat_completion.await_args.args[0]


class TestNoLanguageIsHardcoded:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("named", NAMED_LANGUAGES)
    async def test_the_chapter_prompt_names_no_specific_pair(self, named):
        assert named not in await _main_content_prompt()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("named", NAMED_LANGUAGES)
    async def test_the_bibliographic_prompt_names_no_specific_pair(self, named):
        assert named not in await _bibliographic_prompt()


class TestTheOfficialFormIsStated:
    """The two sections run opposite ways round, and the mẫu says so.

    §1 Nhan đề:  "tiếng Nga (tiếng Anh/tiếng Việt)"  — source language first.
    §2.2 tên chương: "tiếng Việt (tiếng Nga/tiếng Anh)" — Vietnamese first.
    Thuật ngữ:   "tiếng Việt (tiếng Anh)"            — the gloss is English,
                                                       whatever the source is.

    The approved digest follows §1 exactly: "Advances in Adaptive Radar
    Detection and Range Estimation (Những tiến bộ trong phát hiện radar thích
    ứng và ước lượng tầm xa)".
    """

    @pytest.mark.asyncio
    async def test_terms_are_glossed_in_english_whatever_the_source(self):
        prompt = (await _main_content_prompt()).lower()

        assert "english term in parentheses" in prompt
        assert "whatever language the source" in prompt

    @pytest.mark.asyncio
    async def test_the_title_keeps_its_own_language_first(self):
        prompt = (await _bibliographic_prompt()).lower()

        assert "own language first" in prompt
        assert "vietnamese translation in parentheses" in prompt
        assert "any source language" in prompt
