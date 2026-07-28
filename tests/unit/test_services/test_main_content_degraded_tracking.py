"""_summarize_chapter should report when it fell back to raw-text, and why:
degraded=True for an LLM runtime failure, raw_passthrough=True when the LLM
was never called because there wasn't enough content to summarize."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestChapterConcurrency:
    @pytest.mark.asyncio
    async def test_chapters_summarised_concurrently_in_order_with_counters(self):
        """Chapters were summarised strictly one-at-a-time; the pipeline's
        longest stage must fan out with bounded concurrency while keeping
        document order and the degraded/raw_passthrough counters exact."""
        from services.main_content_service import MainContentService

        svc = MainContentService()
        in_flight = [0]
        peak = [0]

        async def fake_summarize(llm, node, number, doc_kind="book"):
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
            await asyncio.sleep(0.02)
            in_flight[0] -= 1
            title = node["title"]
            return (
                {"number": number, "title_vi": title, "content": f"tóm tắt {title}"},
                title == "C2",  # one degraded
                title == "C4",  # one raw passthrough
            )

        tree = {
            "title": "Root",
            "children": [
                {"title": f"C{i}", "content": "x" * 300, "children": []} for i in range(1, 6)
            ],
        }

        chapters, degraded, raw = await svc._summarize_chapters(
            llm=MagicMock(),
            nodes=[{"node": ch, "number": i + 1} for i, ch in enumerate(tree["children"])],
            task_id=None,
            summarize=fake_summarize,
        )

        assert [c["title_vi"] for c in chapters] == ["C1", "C2", "C3", "C4", "C5"]
        assert degraded == 1
        assert raw == 1
        assert peak[0] > 1, "chapters still run sequentially"


class TestSummarizeChapterDegradedFlag:
    @pytest.mark.asyncio
    async def test_returns_degraded_true_on_llm_failure(self):
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = AsyncMock()
        llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
        llm.encoding = None
        llm.chat_completion = AsyncMock(side_effect=RuntimeError("LLM hiccup"))

        node = {"title": "Chapter One", "content": "x" * 200}
        chapter, degraded, raw_passthrough = await svc._summarize_chapter(llm, node, 1)

        assert degraded is True
        assert raw_passthrough is False
        assert chapter["content"].startswith("x" * 100)

    @pytest.mark.asyncio
    async def test_language_instruction_repeated_near_generation_point(self):
        """Regression: confirmed live against the real pipeline LLM
        (qwen3.5-9b) that a Vietnamese-output instruction stated only once,
        followed by ~4000 chars of Russian source text, gets ignored -- the
        model mirrors the source language instead. Repeating the instruction
        immediately before the generation point fixes this (verified live);
        this test locks in that the prompt structure isn't accidentally
        reverted to a single, far-back instruction."""
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = AsyncMock()
        llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
        llm.encoding = None
        llm.chat_completion = AsyncMock(return_value="A summary.")

        node = {"title": "Chapter One", "content": "x" * 200}
        await svc._summarize_chapter(llm, node, 1)

        prompt = llm.chat_completion.await_args[0][0]
        lang_instruction = "MUST respond entirely in Vietnamese"
        first_at = prompt.find(lang_instruction)
        last_at = prompt.rfind(lang_instruction)
        assert first_at != -1
        assert last_at != first_at  # appears at least twice
        # The second occurrence must be after the chapter text, close to
        # the end of the prompt (recency, not just "somewhere later").
        content_end = prompt.rfind("x" * 200) + len("x" * 200)
        assert last_at > content_end
        assert len(prompt) - last_at < 200

    @pytest.mark.asyncio
    async def test_returns_degraded_false_on_success(self):
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = AsyncMock()
        llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
        llm.encoding = None
        llm.chat_completion = AsyncMock(return_value="A clean summary.")

        node = {"title": "Chapter One", "content": "x" * 200}
        chapter, degraded, raw_passthrough = await svc._summarize_chapter(llm, node, 1)

        assert degraded is False
        assert raw_passthrough is False
        assert chapter["content"] == "A clean summary."

    @pytest.mark.asyncio
    async def test_short_content_skips_llm_and_flags_raw_passthrough(self):
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = AsyncMock()
        llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
        llm.encoding = None

        node = {"title": "Chapter One", "content": "too short"}
        chapter, degraded, raw_passthrough = await svc._summarize_chapter(llm, node, 1)

        assert degraded is False
        assert raw_passthrough is True
        llm.chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_prompt_forbids_fabrication(self):
        """Regression: DOC_065 (816pg real book) had a title-page node whose
        gathered text was just an author/affiliation listing ("STRUCTURED /
        SIXTH EDITION / ANDREW S. TANENBAUM ... TODD AUSTIN ..."), 394 chars
        -- well past the raw_passthrough floor, so the LLM was asked to write
        "2-5 sentences covering main topics, methods, and findings". With
        nothing substantive to summarize, it invented a wrong book title
        ("Computer Networks" instead of this book's real title) and a wrong
        co-author, evidently recalling Tanenbaum's *other* famous textbook
        from training data instead of admitting there was nothing to
        summarize. Unlike the node-level prompts in summarization_service.py,
        this prompt had no anti-fabrication constraint at all."""
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = AsyncMock()
        llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
        llm.encoding = None
        llm.chat_completion = AsyncMock(return_value="A summary.")

        node = {"title": "Chapter One", "content": "x" * 200}
        await svc._summarize_chapter(llm, node, 1)

        prompt = llm.chat_completion.await_args[0][0]
        lowered = prompt.lower()
        assert "external knowledge" in lowered
        assert "directly supported" in lowered or "verbatim" in lowered

    @pytest.mark.asyncio
    async def test_heading_only_node_pulls_content_from_children(self):
        """Regression: a tree node whose own text is just a short heading but
        whose real body content lives in nested children (confirmed pattern
        on DOC_059: text_len=66, n_children=4) must not raw-passthrough --
        the LLM should see the children's text too."""
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = AsyncMock()
        llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
        llm.encoding = None
        llm.chat_completion = AsyncMock(return_value="A real summary.")

        node = {
            "title": "1.2.1 Short Heading",
            "content": "short",
            "children": [
                {"title": "sub a", "content": "x" * 100},
                {"title": "sub b", "content": "y" * 100},
            ],
        }
        chapter, degraded, raw_passthrough = await svc._summarize_chapter(llm, node, 1)

        assert degraded is False
        assert raw_passthrough is False
        assert chapter["content"] == "A real summary."
        llm.chat_completion.assert_awaited_once()
        prompt = llm.chat_completion.await_args[0][0]
        assert "x" * 100 in prompt
        assert "y" * 100 in prompt

    @pytest.mark.asyncio
    async def test_truly_empty_node_still_falls_back_without_llm(self):
        from services.main_content_service import MainContentService

        svc = MainContentService()
        llm = AsyncMock()
        llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
        llm.encoding = None

        node = {"title": "Empty Heading", "content": "", "children": []}
        chapter, degraded, raw_passthrough = await svc._summarize_chapter(llm, node, 1)

        assert degraded is False
        assert raw_passthrough is True
        llm.chat_completion.assert_not_called()


class TestGatherNodeTextOrdering:
    def test_depth_first_order_keeps_grandchildren_with_their_parent(self):
        """Regression: text must be gathered in document reading order
        (depth-first), not breadth-first -- otherwise a 3-level-deep
        aggregation interleaves unrelated siblings' text before either
        child's own grandchildren, reading incoherently."""
        from services.main_content_service import _gather_node_text

        node = {
            "title": "root",
            "content": "",
            "children": [
                {
                    "title": "child A",
                    "content": "A-own",
                    "children": [{"title": "A.1", "content": "A-grandchild"}],
                },
                {
                    "title": "child B",
                    "content": "B-own",
                    "children": [{"title": "B.1", "content": "B-grandchild"}],
                },
            ],
        }
        result = _gather_node_text(node, max_chars=10_000)
        parts = result.split("\n\n")

        assert parts.index("A-own") < parts.index("A-grandchild") < parts.index("B-own")
        assert parts.index("B-own") < parts.index("B-grandchild")

    def test_stops_at_char_budget(self):
        """Soft cap: the budget is checked before each node is added, so it
        can overshoot by up to one node's length -- but must stop adding
        further nodes once already over budget."""
        from services.main_content_service import _gather_node_text

        node = {
            "title": "root",
            "content": "x" * 50,
            "children": [
                {"title": "c1", "content": "y" * 50},
                {"title": "c2", "content": "z" * 50},
            ],
        }
        result = _gather_node_text(node, max_chars=60)
        assert "x" * 50 in result
        assert "y" * 50 in result
        assert "z" * 50 not in result

    def test_stops_at_node_visit_cap_even_with_sparse_content(self):
        """Regression: a subtree of mostly-empty nodes must not be walked in
        full just because the char budget never fills up (confirmed real on
        a 761-page document: a single chapter's subtree had 258 nodes)."""
        from services.main_content_service import _gather_node_text

        # 1000 empty children, only the last one has real text -- with a
        # visit cap, the walk gives up long before reaching it.
        empty_children = [{"title": f"empty {i}", "content": ""} for i in range(999)]
        node = {
            "title": "root",
            "content": "",
            "children": empty_children + [{"title": "last", "content": "findable"}],
        }
        result = _gather_node_text(node, max_chars=10_000, max_nodes=50)
        assert "findable" not in result
