"""§2.1 must describe the structure §2.2 lists: Parts and every chapter, 1 to 19.

The abstract used to be composed from the tree's top-level nodes, whose levels come
from a percentile cut — not the chapter units §2.2 uses — so the two sections could
describe different books. The tester's complaint: «chưa xuôi theo một bản tổng thuật».
"""

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.summarization_service import SUMMARY_FORM_EXAMPLE, chapter_outline

FIXTURE = Path(__file__).parents[2] / "fixtures" / "trees" / "ballistics_headings.json"


def _units():
    def unit(n, title, part, summary="Nội dung."):
        return {
            "title": title,
            "chapter_ordinal": n,
            "summary": summary,
            **({"part": part} if part else {}),
        }

    p1 = {"ordinal": 1, "name": "Interior Ballistics"}
    p2 = {"ordinal": 2, "name": "Exterior Ballistics"}
    return [
        unit(1, "Chapter 1 Introductory Concepts", p1),
        unit(2, "Physical Foundation", p1),
        unit(3, "Trajectories", p2, summary=""),
    ]


class TestChapterOutline:
    def test_groups_chapters_under_their_part_with_the_chapter_range(self):
        out = chapter_outline(_units())

        assert "Phần I - Interior Ballistics (chương 1 đến chương 2)" in out
        assert "Phần II - Exterior Ballistics (chương 3)" in out

    def test_lists_every_chapter_even_when_it_has_no_summary_of_its_own(self):
        out = chapter_outline(_units())

        assert "Chương 1. Introductory Concepts: Nội dung." in out
        assert "Chương 3. Trajectories" in out

    def test_a_book_without_parts_is_a_plain_chapter_list(self):
        units = [{k: v for k, v in u.items() if k != "part"} for u in _units()]

        out = chapter_outline(units)

        assert "Phần" not in out and "Chương 2. Physical Foundation" in out


class TestFormExample:
    def test_the_example_is_not_the_testers_answer(self):
        """It was a verbatim copy of the Ballistics reference — it contaminated the
        evaluation and invited the model to copy the content."""
        assert "Đạn đạo" not in SUMMARY_FORM_EXAMPLE
        assert "Phần I" in SUMMARY_FORM_EXAMPLE  # still shows the form


class TestComposeUsesTheUnits:
    @pytest.mark.asyncio
    async def test_the_prompt_walks_all_nineteen_chapters_in_three_parts(self):
        from services.summarization_service import compose_document_summary

        tree = copy.deepcopy(json.loads(FIXTURE.read_text(encoding="utf-8")))

        def label(node):
            node["summary"] = f"Tóm tắt của {node['title']}."
            for child in node.get("children") or []:
                label(child)

        label(tree)
        llm = MagicMock()
        llm.chat_completion = AsyncMock(return_value="Tài liệu gồm 3 phần.")
        llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
        llm.encoding = None

        await compose_document_summary(llm, tree)

        prompt = llm.chat_completion.await_args.args[0]
        assert "Chương 19. Wound Ballistics" in prompt
        assert "Phần III - Terminal Ballistics (chương 14 đến chương 19)" in prompt
        assert all(f"Chương {n}." in prompt for n in range(1, 20))

    @pytest.mark.asyncio
    async def test_a_chapter_without_a_summary_is_never_reported_as_missing(self):
        """Live regression (DOC_011): the abstract said «7 chương chính (trừ chương 8 không
        có nội dung tóm tắt)» because the outline listed chapter 8 with no summary."""
        from services.summarization_service import compose_document_summary

        tree = copy.deepcopy(json.loads(FIXTURE.read_text(encoding="utf-8")))

        def label(node):
            node["summary"] = f"Tóm tắt của {node['title']}."
            for child in node.get("children") or []:
                label(child)

        label(tree)
        llm = MagicMock()
        llm.chat_completion = AsyncMock(return_value="Tài liệu gồm 3 phần.")
        llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
        llm.encoding = None

        await compose_document_summary(llm, tree)

        prompt = llm.chat_completion.await_args.args[0]
        assert "never say that a chapter or its summary is missing" in prompt

    @pytest.mark.asyncio
    async def test_the_prompt_states_the_chapter_count_the_model_must_not_miscount(self):
        """Live regression (DOC_019): a 17-chapter book opened «Tài liệu bao gồm 11 chương»."""
        from services.summarization_service import compose_document_summary

        tree = copy.deepcopy(json.loads(FIXTURE.read_text(encoding="utf-8")))

        def label(node):
            node["summary"] = f"Tóm tắt của {node['title']}."
            for child in node.get("children") or []:
                label(child)

        label(tree)
        llm = MagicMock()
        llm.chat_completion = AsyncMock(return_value="Tài liệu gồm 3 phần.")
        llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
        llm.encoding = None

        await compose_document_summary(llm, tree)

        prompt = llm.chat_completion.await_args.args[0]
        assert "The document has exactly 19 chapters" in prompt
