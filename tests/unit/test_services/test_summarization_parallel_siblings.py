"""Sibling tree nodes should be summarised concurrently, not one at a time."""
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestSiblingSummarizationConcurrency:
    @pytest.mark.asyncio
    async def test_siblings_summarised_concurrently_and_in_order(self):
        from services.summarization_service import SummarizationService

        svc = SummarizationService()

        in_flight = []
        peak = [0]
        # Slowest child listed first — if siblings ran sequentially, its slot
        # would finish long before the others even start.
        delays = {"Child A": 0.03, "Child B": 0.01, "Child C": 0.01}

        async def fake_chat_completion(prompt: str) -> str:
            title = next((t for t in delays if t in prompt), "Child A")
            in_flight.append(title)
            peak[0] = max(peak[0], len(in_flight))
            await asyncio.sleep(delays[title])
            in_flight.remove(title)
            return f"summary-of-{title}"

        llm = AsyncMock()
        llm.chat_completion = AsyncMock(side_effect=fake_chat_completion)

        tree = {
            "title": "Root",
            "content": "",
            "children": [
                {"title": "Child A", "content": "text A", "children": []},
                {"title": "Child B", "content": "text B", "children": []},
                {"title": "Child C", "content": "text C", "children": []},
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

        assert peak[0] >= 2, "siblings should overlap, not run strictly one at a time"

        # Root's synthesis input must still list children in their original order.
        root_prompt = llm.chat_completion.call_args_list[-1].args[0]
        assert root_prompt.index("Child A") < root_prompt.index("Child B") < root_prompt.index("Child C")
        assert meta["nodes_summarised"] == 4
