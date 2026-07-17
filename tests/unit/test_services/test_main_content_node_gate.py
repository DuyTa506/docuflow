"""Node-classification gate for digest §2.2 (main content).

Regression context (DOC_066, a 106-chapter book): the tree faithfully mirrors
every heading, so publisher front matter ("Why Subscribe?", "Customer
support", ...) became ~40 sibling nodes of real chapters. Each rendered as
either a raw title echo or an LLM paragraph saying "this is front matter,
nothing to summarize" — ~40% of §2.2 was structured noise.

The gate classifies all chapter nodes in one batched LLM pass
(substantive / front_matter / toc_fragment), then collapses *consecutive*
non-substantive nodes into a single digest line listing their titles.
Design invariants under test:
- evidence-anchored: `toc_fragment` is only accepted when the node's real
  content is thin (enforced in code, not trusted from the LLM);
- fail-open: any gate failure labels everything substantive (today's
  behavior) and flags `gate_degraded`;
- lossless titles: a merged entry lists every collapsed title.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from services.main_content_service import MainContentService


def _make_nodes(specs):
    """specs: list of (title, content) → chapter work items."""
    return [
        {"node": {"title": title, "content": content, "children": []}, "number": i + 1}
        for i, (title, content) in enumerate(specs)
    ]


def _llm_returning(labels):
    """LLM mock whose chat_completion succeeds and extract_json yields labels."""
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="[]")
    llm.extract_json = MagicMock(return_value=labels)
    return llm


class TestClassifyNodes:
    @pytest.mark.asyncio
    async def test_labels_parsed_from_llm_json(self):
        svc = MainContentService()
        llm = _llm_returning(
            [
                {"number": 1, "label": "front_matter"},
                {"number": 2, "label": "substantive"},
            ]
        )
        nodes = _make_nodes([("Why Subscribe?", "x" * 50), ("Chapter 1", "y" * 500)])

        labels, degraded = await svc._classify_nodes(llm, nodes)

        assert labels == {1: "front_matter", 2: "substantive"}
        assert degraded is False

    @pytest.mark.asyncio
    async def test_llm_failure_defaults_all_substantive_and_flags_degraded(self):
        svc = MainContentService()
        llm = MagicMock()
        llm.chat_completion = AsyncMock(side_effect=RuntimeError("LLM down"))
        nodes = _make_nodes([("A", "x" * 50), ("B", "y" * 500)])

        labels, degraded = await svc._classify_nodes(llm, nodes)

        assert labels == {1: "substantive", 2: "substantive"}
        assert degraded is True

    @pytest.mark.asyncio
    async def test_unparseable_json_defaults_substantive_and_flags_degraded(self):
        svc = MainContentService()
        llm = _llm_returning([])  # _extract_json returns [] on parse failure
        nodes = _make_nodes([("A", "x" * 50)] * 2)

        labels, degraded = await svc._classify_nodes(llm, nodes)

        assert labels == {1: "substantive", 2: "substantive"}
        assert degraded is True

    @pytest.mark.asyncio
    async def test_invalid_rows_are_ignored(self):
        svc = MainContentService()
        llm = _llm_returning(
            [
                {"number": 99, "label": "front_matter"},  # unknown number
                {"number": 1, "label": "banana"},  # unknown label
                {"number": 2, "label": "front_matter"},  # valid
            ]
        )
        nodes = _make_nodes([("A", "x" * 500), ("B", "y" * 50)])

        labels, degraded = await svc._classify_nodes(llm, nodes)

        assert labels == {1: "substantive", 2: "front_matter"}
        assert degraded is False

    @pytest.mark.asyncio
    async def test_toc_fragment_label_requires_thin_content(self):
        """The LLM's `toc_fragment` verdict is only accepted when the node's
        gathered content is actually thin — a fat chapter mislabeled as a
        fragment must be upgraded back to substantive by code."""
        svc = MainContentService()
        llm = _llm_returning(
            [
                {"number": 1, "label": "toc_fragment"},  # thin → accepted
                {"number": 2, "label": "toc_fragment"},  # fat → upgraded
            ]
        )
        nodes = _make_nodes([("Bare heading", "x" * 20), ("Real chapter", "y" * 5000)])

        labels, _ = await svc._classify_nodes(llm, nodes)

        assert labels[1] == "toc_fragment"
        assert labels[2] == "substantive"

    @pytest.mark.asyncio
    async def test_large_node_lists_are_batched(self):
        """106-chapter books must not push the whole listing into one prompt
        near the context limit — the gate batches its LLM calls."""
        svc = MainContentService()
        llm = _llm_returning([])
        llm.extract_json = MagicMock(
            side_effect=lambda _resp: [{"number": n, "label": "substantive"} for n in range(1, 71)]
        )
        nodes = _make_nodes([(f"C{i}", "x" * 200) for i in range(1, 71)])

        await svc._classify_nodes(llm, nodes)

        assert llm.chat_completion.await_count > 1

    @pytest.mark.asyncio
    async def test_prompt_contains_title_length_and_excerpt(self):
        svc = MainContentService()
        llm = _llm_returning([{"number": 1, "label": "substantive"}])
        nodes = _make_nodes([("Distinctive Title", "unique-excerpt-text " * 10)])

        await svc._classify_nodes(llm, nodes)

        prompt = llm.chat_completion.await_args[0][0]
        assert "Distinctive Title" in prompt
        assert "unique-excerpt-text" in prompt
        # unsure → substantive must be an explicit instruction
        assert "substantive" in prompt


class TestSummarizeWithGate:
    def _svc_with_fakes(self, labels, gate_degraded=False):
        svc = MainContentService()
        svc._classify_nodes = AsyncMock(return_value=(labels, gate_degraded))

        async def fake_summarize_chapters(llm, nodes, task_id):
            chapters = [
                {
                    "number": item["number"],
                    "title_vi": item["node"]["title"],
                    "title_original": item["node"]["title"],
                    "content": f"tóm tắt {item['node']['title']}",
                }
                for item in nodes
            ]
            return chapters, 0, 0

        svc._summarize_chapters = fake_summarize_chapters
        return svc

    @pytest.mark.asyncio
    async def test_consecutive_auxiliary_nodes_collapse_into_one_entry(self):
        nodes = _make_nodes(
            [
                ("Why Subscribe?", "x" * 50),
                ("Customer support", "y" * 50),
                ("Chapter One", "z" * 500),
                ("Downloading the example code", "w" * 50),
                ("Chapter Two", "v" * 500),
            ]
        )
        labels = {
            1: "front_matter",
            2: "front_matter",
            3: "substantive",
            4: "toc_fragment",
            5: "substantive",
        }
        svc = self._svc_with_fakes(labels)

        chapters, degraded, raw, aux_sections, gate_degraded = await svc._summarize_with_gate(
            MagicMock(), nodes, task_id=None
        )

        assert len(chapters) == 4
        assert [c["number"] for c in chapters] == [1, 2, 3, 4]
        # merged front-matter run keeps every collapsed title, in order
        assert "Why Subscribe?" in chapters[0]["content"]
        assert "Customer support" in chapters[0]["content"]
        # real chapters summarized with their FINAL numbers
        assert chapters[1]["title_vi"] == "Chapter One"
        assert chapters[1]["number"] == 2
        # isolated mid-book fragment collapses alone
        assert "Downloading the example code" in chapters[2]["content"]
        assert chapters[3]["title_vi"] == "Chapter Two"
        assert aux_sections == 3
        assert gate_degraded is False

    @pytest.mark.asyncio
    async def test_all_substantive_keeps_current_behavior(self):
        nodes = _make_nodes([(f"C{i}", "x" * 500) for i in range(1, 4)])
        labels = {1: "substantive", 2: "substantive", 3: "substantive"}
        svc = self._svc_with_fakes(labels)

        chapters, _, _, aux_sections, gate_degraded = await svc._summarize_with_gate(
            MagicMock(), nodes, task_id=None
        )

        assert [c["title_vi"] for c in chapters] == ["C1", "C2", "C3"]
        assert aux_sections == 0
        assert gate_degraded is False

    @pytest.mark.asyncio
    async def test_gate_degraded_flag_propagates(self):
        nodes = _make_nodes([("A", "x" * 500), ("B", "y" * 500)])
        labels = {1: "substantive", 2: "substantive"}
        svc = self._svc_with_fakes(labels, gate_degraded=True)

        *_, gate_degraded = await svc._summarize_with_gate(MagicMock(), nodes, task_id=None)

        assert gate_degraded is True

    @pytest.mark.asyncio
    async def test_single_node_skips_gate_entirely(self):
        svc = self._svc_with_fakes({1: "substantive"})
        nodes = _make_nodes([("Only chapter", "x" * 500)])

        chapters, *_ = await svc._summarize_with_gate(MagicMock(), nodes, task_id=None)

        svc._classify_nodes.assert_not_awaited()
        assert chapters[0]["title_vi"] == "Only chapter"
