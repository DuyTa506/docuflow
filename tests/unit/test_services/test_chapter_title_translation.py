"""§2.2 in ra `Chương 1. Глава 1. Введение.` trên tài liệu Nga thật (N4.11.160).

Hai lỗi chồng nhau: nhãn cấu trúc bị in hai lần (một Việt một Nga), và tiêu đề
chương chưa bao giờ được dịch — `_summarize_chapter` chỉ tách `"Tên (Gốc)"` bằng
regex ngoặc đơn, mà tiêu đề gốc thì không có ngoặc nào.

Mẫu đòi: `Chương 1. Giới thiệu (Введение).`

Cách làm: **một lượt gọi gộp cho toàn bộ tiêu đề**, không nhét vào lượt tóm tắt.
Đo trên N4.11.160: bắt model vừa viết 150 từ tóm tắt vừa trả JSON đúng định dạng
thì 3/12 chương trả về prose thuần và mất bản dịch. Dịch tiêu đề là việc ngắn,
tách riêng thì nó chỉ phải làm đúng một việc — và lượt tóm tắt quay về prose,
không còn chỗ nào để parse hỏng.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.main_content_service import MainContentService
from utils.digest_format import chapter_heading

_LONG = "Содержание главы о вычислительной технике. " * 40


def _llm(payload):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(
        return_value=payload if isinstance(payload, str) else json.dumps(payload)
    )
    llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
    llm.encoding = None
    # `_parse_json_list` delegates to the client's own extractor; a MagicMock
    # would hand back a MagicMock and every parse would silently "succeed".
    llm.extract_json = MagicMock(side_effect=_extract_json)
    return llm


def _extract_json(text, **kwargs):
    import re

    match = re.search(r"[\[{].*[\]}]", str(text), re.DOTALL)
    if not match:
        raise ValueError("no JSON")
    return json.loads(match.group(0))


def _node(title):
    return {"title": title, "content": "", "children": [{"content": _LONG, "children": []}]}


class TestPrefixStripping:
    """`_summarize_chapter` chỉ lo phần cấu trúc; bản dịch đến ở bước sau."""

    @pytest.mark.asyncio
    async def test_structural_prefix_leaves_the_title(self):
        llm = _llm("Tóm tắt chương.")

        chapter, _, _ = await MainContentService()._summarize_chapter(
            llm, _node("Глава 1. Введение"), 1
        )

        assert chapter["title_original"] == "Введение"
        assert chapter["heading_kind"] == "chapter"
        assert chapter["heading_ordinal"] == 1

    @pytest.mark.asyncio
    async def test_summary_call_stays_plain_prose(self):
        """Không đòi JSON ở lượt tóm tắt — đó là chỗ đã hỏng 3/12 lần."""
        llm = _llm("Tóm tắt chương.")

        chapter, _, _ = await MainContentService()._summarize_chapter(
            llm, _node("Глава 1. Введение"), 1
        )

        assert chapter["content"] == "Tóm tắt chương."
        assert "JSON" not in llm.chat_completion.await_args[0][0]

    @pytest.mark.asyncio
    async def test_a_title_already_in_the_bilingual_form_is_left_alone(self):
        llm = _llm("s")

        chapter, _, _ = await MainContentService()._summarize_chapter(
            llm, _node("Giới thiệu (Introduction)"), 1
        )

        assert (chapter["title_vi"], chapter["title_original"]) == ("Giới thiệu", "Introduction")

    @pytest.mark.asyncio
    async def test_appendix_is_recognised_as_such(self):
        llm = _llm("s")

        chapter, _, _ = await MainContentService()._summarize_chapter(
            llm, _node("Приложение А. Двоичные числа"), 10
        )

        assert (chapter["heading_kind"], chapter["heading_ordinal"]) == ("appendix", 1)


class TestBatchTitleTranslation:
    @pytest.mark.asyncio
    async def test_all_titles_translated_in_one_call(self):
        llm = _llm([{"n": 1, "title_vi": "Giới thiệu"}, {"n": 2, "title_vi": "Tầng logic số"}])
        chapters = [
            {"number": 1, "title_vi": "Введение", "title_original": "Введение"},
            {
                "number": 2,
                "title_vi": "Цифровой логический уровень",
                "title_original": "Цифровой логический уровень",
            },
        ]

        await MainContentService()._translate_titles(llm, chapters)

        assert [c["title_vi"] for c in chapters] == ["Giới thiệu", "Tầng logic số"]
        assert llm.chat_completion.await_count == 1

    @pytest.mark.asyncio
    async def test_a_label_the_model_prepends_is_stripped(self):
        """Quan sát thật: model trả «Phụ lục B. Số thực» → in ra «Phụ lục B. Phụ lục B. …»."""
        llm = _llm([{"n": 1, "title_vi": "Phụ lục B. Số thực và chuẩn IEEE 754"}])
        chapters = [{"number": 1, "title_vi": "x", "title_original": "x"}]

        await MainContentService()._translate_titles(llm, chapters)

        assert chapters[0]["title_vi"] == "Số thực và chuẩn IEEE 754"

    @pytest.mark.asyncio
    async def test_untranslated_entries_keep_the_original(self):
        llm = _llm([{"n": 1, "title_vi": "Giới thiệu"}])
        chapters = [
            {"number": 1, "title_vi": "Введение", "title_original": "Введение"},
            {"number": 2, "title_vi": "Библиография", "title_original": "Библиография"},
        ]

        await MainContentService()._translate_titles(llm, chapters)

        assert chapters[1]["title_vi"] == "Библиография"

    @pytest.mark.asyncio
    async def test_llm_failure_leaves_every_title_untouched(self):
        llm = _llm("")
        llm.chat_completion = AsyncMock(side_effect=RuntimeError("down"))
        chapters = [{"number": 1, "title_vi": "Введение", "title_original": "Введение"}]

        await MainContentService()._translate_titles(llm, chapters)

        assert chapters[0]["title_vi"] == "Введение"

    @pytest.mark.asyncio
    async def test_prose_answer_is_ignored_rather_than_stored(self):
        llm = _llm("Tôi đã dịch các tiêu đề rồi nhé.")
        chapters = [{"number": 1, "title_vi": "Введение", "title_original": "Введение"}]

        await MainContentService()._translate_titles(llm, chapters)

        assert chapters[0]["title_vi"] == "Введение"


class TestNumberedOriginalTitles:
    """Live regression (DOC_008, DOC_011): original titles already carried their
    own number, so the listing read ``1. 2. What is a Digital Twin?``; the model
    answered with the title's number and every title_vi shifted by one."""

    @staticmethod
    def _llm_reading_nearest_number():
        import re

        async def _answer(prompt, **kwargs):
            listing = prompt.split("TITLES:\n", 1)[1].split("\n\n", 1)[0]
            rows = []
            for line in listing.splitlines():
                # A model keys on the number sitting right before the words.
                m = re.search(r"(\d+)[.\]]\s+([^\d\s][^\n]*)$", line)
                rows.append({"n": int(m.group(1)), "title_vi": f"VI:{m.group(2)}"})
            return json.dumps(rows)

        llm = _llm([])
        llm.chat_completion = AsyncMock(side_effect=_answer)
        return llm

    @pytest.mark.asyncio
    async def test_titles_stay_aligned_with_their_chapter(self):
        chapters = [
            {"number": 1, "title_vi": "", "title_original": "2. What is a Digital Twin?"},
            {"number": 2, "title_vi": "", "title_original": "3. Digital Twin Architecture"},
            {"number": 3, "title_vi": "", "title_original": "4. Applications"},
        ]

        await MainContentService()._translate_titles(self._llm_reading_nearest_number(), chapters)

        assert [c["title_vi"] for c in chapters] == [
            "VI:What is a Digital Twin?",
            "VI:Digital Twin Architecture",
            "VI:Applications",
        ]


class TestDoclingNumberedChapterHeadings:
    """Live regression (DOC_013): Docling headings read ``6 Chapter 6: Thread…``.
    The stray leading number hid the ``Chapter 6`` label, the listing sent
    ``[5] Chapter 6: …``, the model answered ``n: 6`` and title_vi landed on
    the wrong chapter ("Các lệnh Regex ⟵ Chapter 15: Shared Libraries")."""

    @pytest.mark.asyncio
    async def test_leading_number_before_the_label_is_stripped(self):
        llm = _llm("Tóm tắt chương.")

        chapter, _, _ = await MainContentService()._summarize_chapter(
            llm, _node("6 Chapter 6: Thread, Frame & Stepping Around"), 5
        )

        assert chapter["title_original"] == "Thread, Frame & Stepping Around"
        assert chapter["heading_kind"] == "chapter"
        assert chapter["heading_ordinal"] == 6

    @pytest.mark.asyncio
    async def test_unit_chapter_number_is_kept_when_the_title_has_none(self):
        llm = _llm("Tóm tắt chương.")
        node = {**_node("Trajectories"), "chapter_ordinal": 7}

        chapter, _, _ = await MainContentService()._summarize_chapter(llm, node, 6)

        assert chapter["title_original"] == "Trajectories"
        assert (chapter["heading_kind"], chapter["heading_ordinal"]) == ("chapter", 7)

    @pytest.mark.asyncio
    async def test_listing_never_shows_a_chapter_number(self):
        chapters = [
            {"number": 5, "title_vi": "", "title_original": "6 Chapter 6: Thread & Frame"},
            {"number": 6, "title_vi": "", "title_original": "10 Chapter 10: Regex Commands"},
        ]

        await MainContentService()._translate_titles(
            TestNumberedOriginalTitles._llm_reading_nearest_number(), chapters
        )

        assert [c["title_vi"] for c in chapters] == ["VI:Thread & Frame", "VI:Regex Commands"]


class TestRenderedLine:
    def test_chapter_matches_the_official_form(self):
        assert (
            chapter_heading(1, "Giới thiệu", "Введение", heading_kind="chapter", heading_ordinal=1)
            == "Chương 1. Giới thiệu (Введение)."
        )

    def test_appendix_is_not_labelled_as_a_chapter(self):
        assert (
            chapter_heading(
                10, "Số nhị phân", "Двоичные числа", heading_kind="appendix", heading_ordinal=1
            )
            == "Phụ lục A. Số nhị phân (Двоичные числа)."
        )
