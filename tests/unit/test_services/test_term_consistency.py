"""Tạo sao: 13/26 cặp (thuật ngữ, cách viết) của sách Ballistics khớp termbase — «nội đạn học»
cạnh «Đạn đạo trong nòng», «yaw of repose» để nguyên tiếng Anh. Prompt đã chứa termbase
mà model vẫn lệch một nửa số lần, nên có một lượt sửa riêng chỉ đổi cách gọi thuật ngữ."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.term_consistency import enforce_termbase, safe_revision

GLOSSARY = {"Interior Ballistics": "Đạn đạo trong nòng", "Yaw of Repose": "Góc lệch dừng"}
TEXT = (
    "Chương trình bày nội đạn học (interior ballistics) với 3 phương pháp và áp suất 292 psi. "
    "Khái niệm yaw of repose được dùng ở đây."
)


def _llm(reply):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=reply)
    return llm


class TestSafeRevision:
    def test_a_rewrite_that_only_swaps_terms_is_accepted(self):
        revised = TEXT.replace("nội đạn học", "Đạn đạo trong nòng")

        assert safe_revision(TEXT, revised)

    def test_a_changed_number_is_rejected(self):
        assert not safe_revision(TEXT, TEXT.replace("292", "300"))

    def test_a_much_shorter_answer_is_rejected(self):
        assert not safe_revision(TEXT, "Đã sửa.")

    def test_an_empty_answer_is_rejected(self):
        assert not safe_revision(TEXT, "")


class TestEnforceTermbase:
    @pytest.mark.asyncio
    async def test_the_prompt_lists_only_the_terms_that_occur(self):
        llm = _llm(TEXT)

        await enforce_termbase(llm, TEXT, {**GLOSSARY, "Shock Physics": "Vật lý sóng va đập"})

        prompt = llm.chat_completion.await_args.args[0]
        assert "Interior Ballistics → Đạn đạo trong nòng" in prompt
        assert "Shock Physics" not in prompt

    @pytest.mark.asyncio
    async def test_a_safe_rewrite_replaces_the_text(self):
        revised = TEXT.replace("nội đạn học", "Đạn đạo trong nòng").replace(
            "yaw of repose", "góc lệch dừng"
        )

        assert await enforce_termbase(_llm(revised), TEXT, GLOSSARY) == revised

    @pytest.mark.asyncio
    async def test_an_unsafe_rewrite_keeps_the_original(self):
        assert await enforce_termbase(_llm(TEXT.replace("292", "300")), TEXT, GLOSSARY) == TEXT

    @pytest.mark.asyncio
    async def test_a_text_with_none_of_the_terms_costs_no_call(self):
        llm = _llm("x")

        assert await enforce_termbase(llm, "Không có thuật ngữ nào.", GLOSSARY) == (
            "Không có thuật ngữ nào."
        )
        assert llm.chat_completion.await_count == 0

    @pytest.mark.asyncio
    async def test_a_failed_call_keeps_the_original(self):
        llm = MagicMock()
        llm.chat_completion = AsyncMock(side_effect=RuntimeError("down"))

        assert await enforce_termbase(llm, TEXT, GLOSSARY) == TEXT


class TestWiredIntoTheDigest:
    @pytest.mark.asyncio
    async def test_every_chapter_text_goes_through_the_pass(self):
        from unittest.mock import patch

        from services.main_content_service import MainContentService

        chapters = [{"number": 1, "content": TEXT}, {"number": 2, "content": "Không có gì."}]
        fixed = TEXT.replace("nội đạn học", "Đạn đạo trong nòng")

        with patch("services.main_content_service.enforce_termbase", new=AsyncMock(
            side_effect=lambda llm, text, glossary: fixed if text == TEXT else text
        )):
            await MainContentService()._enforce_terms(MagicMock(), chapters, GLOSSARY)

        assert chapters[0]["content"] == fixed
        assert chapters[1]["content"] == "Không có gì."

    @pytest.mark.asyncio
    async def test_an_empty_glossary_costs_nothing(self):
        from services.main_content_service import MainContentService

        llm = _llm("x")
        chapters = [{"number": 1, "content": TEXT}]

        await MainContentService()._enforce_terms(llm, chapters, {})

        assert llm.chat_completion.await_count == 0


class TestTheAbstractGoesThroughThePassToo:
    @pytest.mark.asyncio
    async def test_the_composed_abstract_is_worded_by_the_termbase(self):
        import copy
        import json
        from pathlib import Path
        from unittest.mock import patch

        from services.domain_context import DomainContext, _current
        from services.summarization_service import compose_document_summary

        fixture = Path(__file__).parents[2] / "fixtures" / "trees" / "ballistics_headings.json"
        tree = copy.deepcopy(json.loads(fixture.read_text(encoding="utf-8")))

        def label(node):
            node["summary"] = f"Tóm tắt của {node['title']}."
            for child in node.get("children") or []:
                label(child)

        label(tree)
        _current.set(
            DomainContext({}, {"Interior Ballistics": {"vi": "Đạn đạo trong nòng", "en": "x"}})
        )
        llm = _llm("Tài liệu về nội đạn học (interior ballistics).")
        llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
        llm.encoding = None

        with patch(
            "services.summarization_service.enforce_termbase",
            new=AsyncMock(return_value="Tài liệu về Đạn đạo trong nòng (interior ballistics)."),
        ) as enforce:
            result = await compose_document_summary(llm, tree)

        assert result == "Tài liệu về Đạn đạo trong nòng (interior ballistics)."
        assert enforce.await_args.args[2] == {"Interior Ballistics": "Đạn đạo trong nòng"}
        _current.set(DomainContext())
