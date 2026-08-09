"""Adaptive cluster summarization for very large trees: above a node-count
threshold, leaf siblings are batched into ONE prompt per cluster instead of
one LLM call per node (a 20k-node book meant 20k calls), with per-node
fallback when the batch JSON is unusable, bounded fan-out, and periodic
checkpointing so a retry resumes instead of redoing every node.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _leaf(title, marker):
    return {"title": title, "content": f"{marker} nội dung " * 30, "children": []}


def _tree():
    return {
        "title": "Root",
        "content": "root content",
        "children": [
            {
                "title": "Chương 1",
                "content": "nội dung chương 1",
                "children": [_leaf("1.1", "M11"), _leaf("1.2", "M12"), _leaf("1.3", "M13")],
            },
            {
                "title": "Chương 2",
                "content": "nội dung chương 2",
                "children": [_leaf("2.1", "M21"), _leaf("2.2", "M22")],
            },
        ],
    }


def _make_llm():
    llm = AsyncMock()
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None

    def _extract(response, **kwargs):
        return json.loads(response)

    llm.extract_json = MagicMock(side_effect=_extract)

    async def _completion(prompt, **kwargs):
        if "Return ONLY valid JSON" in prompt and "SECTION n" in prompt:
            n = prompt.count("SECTION n")
            return json.dumps(
                {
                    "summaries": [
                        {"id": f"n{i+1}", "summary": f"tóm tắt cụm {i+1}"} for i in range(n)
                    ]
                }
            )
        return "tóm tắt đơn"

    llm.chat_completion = AsyncMock(side_effect=_completion)
    return llm


def _mock_db(monkeypatch_target, tree):
    fake_tree_index = MagicMock()
    fake_tree_index.tree_data = tree
    fake_tree_index.tree_data_key = None
    fake_tree_index.id = "TI_C"

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
        fake_tree_index
    )
    session.query.return_value.filter.return_value.first.return_value = fake_tree_index

    manager = MagicMock()
    manager.session.return_value = session
    return manager, fake_tree_index


class TestClusterMode:
    @pytest.mark.asyncio
    async def test_leaves_are_batched_per_parent_above_threshold(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()
        tree = _tree()

        with (
            patch("services.summarization_service.get_db_manager") as mock_dbm,
            patch("services.summarization_service.settings") as mock_settings,
        ):
            mock_settings.summarize_cluster_threshold = 3
            mock_settings.summarize_cluster_max_nodes = 10
            mock_settings.summarize_checkpoint_nodes = 1000
            mock_settings.summarize_node_content_tokens = 1500
            mock_settings.ai_max_concurrent_requests = 4
            mock_settings.ai_input_budget_tokens = 10000
            mock_dbm.return_value, _ = _mock_db(None, tree)[0], None
            manager, _ = _mock_db(None, tree)
            mock_dbm.return_value = manager

            summary, meta = await svc._hierarchical_tree_summarize("DOC_C", llm, task_id=None)

        batch_prompts = [
            c.args[0] for c in llm.chat_completion.call_args_list if "SECTION n" in c.args[0]
        ]
        # 5 leaves under 2 parents → 2 batch calls, not 5 single calls
        assert len(batch_prompts) == 2
        # every leaf got its clustered summary mapped back
        for parent in tree["children"]:
            for leaf in parent["children"]:
                assert leaf["summary"].startswith("tóm tắt cụm")
        assert meta["nodes_summarised"] == 8

    @pytest.mark.asyncio
    async def test_malformed_batch_json_falls_back_per_node(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()

        async def _bad_completion(prompt, **kwargs):
            if "SECTION n" in prompt:
                return "not json at all"
            return "tóm tắt đơn"

        llm.chat_completion = AsyncMock(side_effect=_bad_completion)
        llm.extract_json = MagicMock(side_effect=ValueError("bad json"))
        tree = _tree()

        with (
            patch("services.summarization_service.get_db_manager") as mock_dbm,
            patch("services.summarization_service.settings") as mock_settings,
        ):
            mock_settings.summarize_cluster_threshold = 3
            mock_settings.summarize_cluster_max_nodes = 10
            mock_settings.summarize_checkpoint_nodes = 1000
            mock_settings.summarize_node_content_tokens = 1500
            mock_settings.ai_max_concurrent_requests = 4
            mock_settings.ai_input_budget_tokens = 10000
            manager, _ = _mock_db(None, tree)
            mock_dbm.return_value = manager

            await svc._hierarchical_tree_summarize("DOC_C", llm, task_id=None)

        for parent in tree["children"]:
            for leaf in parent["children"]:
                assert leaf["summary"] == "tóm tắt đơn"

    @pytest.mark.asyncio
    async def test_below_threshold_keeps_per_node_calls(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()
        tree = _tree()

        with (
            patch("services.summarization_service.get_db_manager") as mock_dbm,
            patch("services.summarization_service.settings") as mock_settings,
        ):
            mock_settings.summarize_cluster_threshold = 800
            mock_settings.summarize_cluster_max_nodes = 10
            mock_settings.summarize_checkpoint_nodes = 1000
            mock_settings.summarize_node_content_tokens = 1500
            mock_settings.ai_max_concurrent_requests = 4
            mock_settings.ai_input_budget_tokens = 10000
            manager, _ = _mock_db(None, tree)
            mock_dbm.return_value = manager

            await svc._hierarchical_tree_summarize("DOC_C", llm, task_id=None)

        assert not any("SECTION n" in c.args[0] for c in llm.chat_completion.call_args_list)

    @pytest.mark.asyncio
    async def test_resume_skips_nodes_with_existing_summaries(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()
        tree = _tree()
        # first parent's subtree fully summarised in a previous attempt
        tree["children"][0]["summary"] = "đã có"
        for leaf in tree["children"][0]["children"]:
            leaf["summary"] = "đã có"

        with (
            patch("services.summarization_service.get_db_manager") as mock_dbm,
            patch("services.summarization_service.settings") as mock_settings,
        ):
            mock_settings.summarize_cluster_threshold = 800
            mock_settings.summarize_cluster_max_nodes = 10
            mock_settings.summarize_checkpoint_nodes = 1000
            mock_settings.summarize_node_content_tokens = 1500
            mock_settings.ai_max_concurrent_requests = 4
            mock_settings.ai_input_budget_tokens = 10000
            manager, _ = _mock_db(None, tree)
            mock_dbm.return_value = manager

            await svc._hierarchical_tree_summarize("DOC_C", llm, task_id=None)

        prompts = [c.args[0] for c in llm.chat_completion.call_args_list]
        assert not any("M11" in p or "M12" in p or "M13" in p for p in prompts)
        assert any("M21" in p for p in prompts)

    @pytest.mark.asyncio
    async def test_checkpoint_persists_at_cadence(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()
        llm = _make_llm()
        tree = _tree()

        with (
            patch("services.summarization_service.get_db_manager") as mock_dbm,
            patch("services.summarization_service.settings") as mock_settings,
            patch("services.summarization_service._persist_tree") as mock_persist,
        ):
            mock_settings.summarize_cluster_threshold = 800
            mock_settings.summarize_cluster_max_nodes = 10
            mock_settings.summarize_checkpoint_nodes = 2
            mock_settings.summarize_node_content_tokens = 1500
            mock_settings.ai_max_concurrent_requests = 4
            mock_settings.ai_input_budget_tokens = 10000
            manager, _ = _mock_db(None, tree)
            mock_dbm.return_value = manager

            await svc._hierarchical_tree_summarize("DOC_C", llm, task_id=None)

        # 8 nodes, cadence 2 → at least 3 intermediate + 1 final persist
        assert mock_persist.call_count >= 4
