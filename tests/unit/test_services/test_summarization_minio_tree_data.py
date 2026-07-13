"""Large documents offload TreeIndex.tree_data to MinIO (tree_data_key) and
leave the tree_data JSON column NULL (see utils/tree_payload.py). Summarize
must use get_tree_payload() to resolve that, not read tree_index.tree_data
directly — otherwise dict(None) raises TypeError and the digest pipeline gets
stuck (see workflows/digest_workflow.py's failure-handling bug, fixed
alongside this).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_llm(response="Summary text"):
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value=response)
    llm.count_tokens = MagicMock(return_value=10)
    return llm


@pytest.mark.asyncio
async def test_hierarchical_summarize_falls_back_to_minio_tree_data():
    from services.summarization_service import SummarizationService

    svc = SummarizationService()
    llm = _make_llm()

    tree = {
        "title": "Root",
        "content": "root content",
        "children": [{"title": "Child", "content": "child content", "children": []}],
    }

    fake_tree_index = MagicMock()
    fake_tree_index.tree_data = None
    fake_tree_index.tree_data_key = "docs/DOC_001/tree.json"
    fake_tree_index.id = "TI_001"

    fake_storage = MagicMock()
    fake_storage.exists.return_value = True
    fake_storage.get_bytes.return_value = json.dumps(tree).encode("utf-8")

    with (
        patch("services.summarization_service.get_db_manager") as mock_dbm,
        patch("utils.tree_payload.get_object_storage", return_value=fake_storage),
    ):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
            fake_tree_index
        )
        mock_dbm.return_value.session.return_value = mock_session

        document_summary, meta = await svc._hierarchical_tree_summarize(
            "DOC_001", llm, task_id=None
        )

    assert document_summary
    fake_storage.get_bytes.assert_called_once_with("docs/DOC_001/tree.json")


@pytest.mark.asyncio
async def test_offloaded_tree_writes_summaries_back_to_minio_not_column():
    """A MinIO-offloaded tree must be checkpointed back to MinIO — writing the
    JSON column would re-bloat what the offload migration removed."""
    from services.summarization_service import SummarizationService

    svc = SummarizationService()
    llm = _make_llm()
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None

    tree = {
        "title": "Root",
        "content": "root content",
        "children": [{"title": "Child", "content": "child content", "children": []}],
    }

    fake_tree_index = MagicMock()
    fake_tree_index.tree_data = None
    fake_tree_index.tree_data_key = "docs/DOC_001/tree.json"
    fake_tree_index.id = "TI_001"

    read_storage = MagicMock()
    read_storage.exists.return_value = True
    read_storage.get_bytes.return_value = json.dumps(tree).encode("utf-8")

    write_storage = MagicMock()

    with (
        patch("services.summarization_service.get_db_manager") as mock_dbm,
        patch("utils.tree_payload.get_object_storage", return_value=read_storage),
        patch("services.object_storage.get_object_storage", return_value=write_storage),
    ):
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
            fake_tree_index
        )
        mock_session.query.return_value.filter.return_value.first.return_value = fake_tree_index
        mock_dbm.return_value.session.return_value = mock_session

        await svc._hierarchical_tree_summarize("DOC_001", llm, task_id=None)

    # Final checkpoint went to MinIO under the same key…
    write_calls = [c for c in write_storage.put_bytes.call_args_list]
    assert write_calls, "expected tree checkpoint written to object storage"
    assert write_calls[-1].args[0] == "docs/DOC_001/tree.json"
    # …and the JSON column was never repopulated.
    assert fake_tree_index.tree_data is None
