"""§2.1 "Tóm tắt" exported the literal placeholder `[Chưa có — chạy summarize trước]`.

The digest reads its abstract from the *root* node's summary, but the level walk
summarises the nodes inside the tree — the synthetic root wrapper is not one of
them, so `tree_data["summary"]` stayed empty. Meanwhile every one of the
thousands of node summaries the stage had just paid for was checkpointed into
the tree and never read again.

Observed on N4.11.160: the digest listed 265 chapter summaries under §2.2 while
§2.1 showed the placeholder.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.summarization_service import compose_document_summary


def _llm(response="Tổng quan tài liệu."):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=response)
    llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
    llm.encoding = None
    return llm


@pytest.mark.asyncio
async def test_existing_root_summary_is_used_when_nothing_to_compose_from():
    llm = _llm()
    tree = {"title": "Doc", "summary": "Bản tóm tắt gốc.", "children": []}

    assert await compose_document_summary(llm, tree) == "Bản tóm tắt gốc."
    llm.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_level_walk_root_summary_is_recomposed_in_the_digest_form():
    """The level walk summarises the root too, with the generic section prompt.

    Returning that as-is meant the §2.1 prompt below never ran: DOC_136 got the
    generic synthesis ("Bản phân tích này xem xét…", author names included)
    instead of the organisation-first form the library asked for.
    """
    llm = _llm("Tài liệu gồm 2 chương.")
    tree = {
        "title": "Doc",
        "summary": "Bản tổng hợp chung của bước duyệt cây.",
        "children": [
            {"title": "Глава 1", "summary": "Nội dung chương một.", "children": []},
            {"title": "Глава 2", "summary": "Nội dung chương hai.", "children": []},
        ],
    }

    assert await compose_document_summary(llm, tree) == "Tài liệu gồm 2 chương."
    llm.chat_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_the_existing_root_summary():
    llm = _llm()
    llm.chat_completion = AsyncMock(side_effect=RuntimeError("LLM down"))
    tree = {
        "title": "Doc",
        "summary": "Bản tổng hợp chung của bước duyệt cây.",
        "children": [{"title": "Глава 1", "summary": "Nội dung chương một.", "children": []}],
    }

    assert await compose_document_summary(llm, tree) == "Bản tổng hợp chung của bước duyệt cây."


@pytest.mark.asyncio
async def test_root_summary_is_composed_from_top_level_node_summaries():
    llm = _llm()
    tree = {
        "title": "Архитектура компьютера",
        "children": [
            {"title": f"Глава {i}", "summary": f"Tóm tắt chương {i}.", "children": []}
            for i in range(1, 10)
        ],
    }

    result = await compose_document_summary(llm, tree)

    assert result == "Tổng quan tài liệu."
    prompt = llm.chat_completion.await_args.args[0]
    assert "Tóm tắt chương 1." in prompt
    assert "Tóm tắt chương 9." in prompt, "later chapters were dropped from the composition"


@pytest.mark.asyncio
async def test_returns_empty_when_nothing_to_compose_from():
    llm = _llm()
    tree = {"title": "Doc", "children": [{"title": "Section", "children": []}]}

    assert await compose_document_summary(llm, tree) == ""
    llm.chat_completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_failure_degrades_to_concatenated_node_summaries():
    """A failed composition must not resurrect the placeholder."""
    llm = _llm()
    llm.chat_completion = AsyncMock(side_effect=RuntimeError("LLM down"))
    tree = {
        "title": "Doc",
        "children": [
            {"title": "Глава 1", "summary": "Nội dung chương một.", "children": []},
            {"title": "Глава 2", "summary": "Nội dung chương hai.", "children": []},
        ],
    }

    result = await compose_document_summary(llm, tree)

    assert "Nội dung chương một." in result
    assert "Nội dung chương hai." in result


# ── §2.1 follows the digest form: organisation first, then a walk through it ──
#
# The prompt used to say "Do NOT enumerate the sections". A tester rejected
# the resulting abstract as "chưa xuôi theo một bản tổng thuật tài
# liệu" and wrote the form expected: "Tài liệu bao gồm 3 phần:
# Phần I - Đạn đạo trong (bao gồm từ chương 1 đến chương 5),
# trình bày…; Phần II…" — the organisation, then each part in order.


def _book_with_parts():
    return {
        "title": "Ballistics",
        "children": [
            {
                "title": f"Part {p}",
                "summary": f"Tóm tắt phần {p}.",
                "children": [
                    {"title": f"Chapter {c}", "summary": f"Tóm tắt chương {c}.", "children": []}
                    for c in range(3 * p - 2, 3 * p + 1)
                ],
            }
            for p in (1, 2, 3)
        ],
    }


@pytest.mark.asyncio
async def test_prompt_asks_for_the_organisation_and_a_walk_through_it():
    llm = _llm()

    await compose_document_summary(llm, _book_with_parts())

    prompt = llm.chat_completion.await_args.args[0]
    assert "Do NOT enumerate the sections" not in prompt
    assert "how the document is organised" in prompt
    # The form cue shows the tester's shape (organisation → parts → chapter ranges) on a
    # neutral subject, so the model copies the form and not the Ballistics content.
    assert "Tài liệu bao gồm 2 phần: Phần I" in prompt


@pytest.mark.asyncio
async def test_a_part_lists_its_chapters_so_their_range_can_be_stated():
    llm = _llm()

    await compose_document_summary(llm, _book_with_parts())

    prompt = llm.chat_completion.await_args.args[0]
    assert "Chapter 1" in prompt and "Chapter 3" in prompt
    assert "Chapter 7" in prompt and "Chapter 9" in prompt


@pytest.mark.asyncio
async def test_single_root_article_is_outlined_by_its_sections():
    """A journal article's tree is one root (the title) with the sections under it."""
    llm = _llm()
    tree = {
        "title": "Document",
        "children": [
            {
                "title": "Intelligent Tutors for Adult Learners",
                "summary": "Tóm tắt toàn bài.",
                "children": [
                    {"title": name, "summary": f"Tóm tắt mục {name}.", "children": []}
                    for name in ("Introduction", "Study 1", "Study 2", "Discussion")
                ],
            }
        ],
    }

    await compose_document_summary(llm, tree)

    prompt = llm.chat_completion.await_args.args[0]
    for name in ("Introduction", "Study 1", "Study 2", "Discussion"):
        assert f"Tóm tắt mục {name}." in prompt


@pytest.mark.parametrize(
    "echo",
    ["2.1 Tóm tắt\n", "2.1. Tóm tắt\n\n", "**2.1. Tóm tắt**\n", "## 2.1 Tóm tắt\n", "Tóm tắt:\n"],
)
@pytest.mark.asyncio
async def test_echoed_section_heading_is_dropped(echo):
    """The prompt names "section 2.1 (Tóm tắt)" and the model echoed it as a first
    line (Nội dung 4, lượt 3: 45/52 abstracts). The digest adds its own heading,
    so the exported §2.1 read "2.1. Tóm tắt / 2.1 Tóm tắt"."""
    llm = _llm(f"{echo}Tài liệu gồm 2 chương.")
    tree = {
        "title": "Doc",
        "children": [
            {"title": "Chương 1", "summary": "Nội dung chương một.", "children": []},
            {"title": "Chương 2", "summary": "Nội dung chương hai.", "children": []},
        ],
    }

    assert await compose_document_summary(llm, tree) == "Tài liệu gồm 2 chương."


@pytest.mark.asyncio
async def test_a_sentence_mentioning_the_summary_is_kept():
    llm = _llm("Tóm tắt lại, tài liệu gồm 2 chương.")
    tree = {
        "title": "Doc",
        "children": [
            {"title": "Chương 1", "summary": "Nội dung chương một.", "children": []},
        ],
    }

    assert await compose_document_summary(llm, tree) == "Tóm tắt lại, tài liệu gồm 2 chương."
