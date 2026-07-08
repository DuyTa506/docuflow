"""Per-node LLM failures during hierarchical summarization should be counted,
not silently swallowed into an indistinguishable raw-text fallback."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestDegradedNodeTracking:
    @pytest.mark.asyncio
    async def test_failed_node_is_counted_and_still_produces_fallback_text(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()

        async def fake_chat_completion(prompt: str) -> str:
            # Only the leaf-level call for the broken child should fail — not
            # the root's synthesis prompt, which quotes the child's fallback
            # text and would otherwise match too.
            if "Sub-section summaries" not in prompt and "some content here" in prompt:
                raise RuntimeError("LLM hiccup")
            return "ok summary"

        llm = AsyncMock()
        llm.chat_completion = AsyncMock(side_effect=fake_chat_completion)

        tree = {
            "title": "Root",
            "content": "",
            "children": [
                {"title": "Broken Child", "content": "some content here", "children": []},
                {"title": "Fine Child", "content": "other content here", "children": []},
            ],
        }
        fake_tree_index = MagicMock()
        fake_tree_index.tree_data = tree
        fake_tree_index.id = "TI_001"

        with patch("services.summarization_service.get_db_manager") as mock_dbm:
            mock_session = MagicMock()
            mock_session.__enter__ = MagicMock(return_value=mock_session)
            mock_session.__exit__ = MagicMock(return_value=False)
            mock_session.query.return_value.filter.return_value \
                .order_by.return_value.first.return_value = fake_tree_index
            mock_dbm.return_value.session.return_value = mock_session

            document_summary, meta = await svc._hierarchical_tree_summarize(
                "DOC_001", llm, task_id=None
            )

        assert meta["degraded_nodes"] == 1
        assert meta["nodes_summarised"] == 3
        # The broken node's summary is still present (fallback text), just counted.
        assert "some content here" in tree["children"][0]["summary"]
