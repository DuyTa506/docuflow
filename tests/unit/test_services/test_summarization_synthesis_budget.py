"""Parent/root synthesis assembles ALL of its children's summaries into one
prompt with no token budget applied — unlike leaf/batch prompts, which are
already truncated via enricher.truncate_to_tokens. On a real 816-page book
(816pg-class E2E test, DOC_065) the root node had 265 direct children;
their concatenated summaries alone were ~104k tokens against a 16k context
window, so the LLM synthesis call failed with a context-size-exceeded 400.
The `except` fallback (`own_content[:200]`) is empty for a root wrapper
node with no own text, so the failure was silently swallowed into an EMPTY
document abstract — DOC_063 hit the same bug at ~161k tokens. This isn't
edge-case: any document with enough chapters to exceed context size at the
root level reproduces it.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_llm():
    llm = AsyncMock()
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    llm.chat_completion = AsyncMock(return_value="tóm tắt tổng hợp")
    return llm


def _mock_db(tree):
    fake_tree_index = MagicMock()
    fake_tree_index.tree_data = tree
    fake_tree_index.tree_data_key = None
    fake_tree_index.id = "TI_WIDE"

    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.query.return_value.filter.return_value.order_by.return_value.first.return_value = (
        fake_tree_index
    )
    session.query.return_value.filter.return_value.first.return_value = fake_tree_index

    manager = MagicMock()
    manager.session.return_value = session
    return manager


def _wide_tree(num_children: int = 50) -> dict:
    # Root wrapper node with no own content — matches the real "Document"
    # top-level node — and many children already summarised (simulating a
    # resumed run where every leaf/chapter is done and only the root-level
    # synthesis remains).
    children = [
        {
            "title": f"Chương {i}",
            "content": "",
            "children": [],
            "summary": f"tóm tắt chương {i} nội dung chi tiết " * 60,  # ~2000 chars
        }
        for i in range(num_children)
    ]
    return {"title": "Document", "content": "", "children": children}


@pytest.mark.asyncio
async def test_root_synthesis_prompt_is_budgeted_for_wide_fanout():
    from services.summarization_service import SummarizationService

    svc = SummarizationService()
    llm = _make_llm()
    tree = _wide_tree(num_children=50)

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
        mock_dbm.return_value = _mock_db(tree)

        summary, meta = await svc._hierarchical_tree_summarize("DOC_WIDE", llm, task_id=None)

    root_prompts = [
        c.args[0]
        for c in llm.chat_completion.call_args_list
        if "Sub-section summaries" in c.args[0]
    ]
    assert len(root_prompts) == 1
    # Raw concatenation of all 50 child summaries would be ~100k chars —
    # budget = max(1000, ai_input_budget_tokens - 1500) tokens, ~4 chars/token
    # fallback in the test's mocked (non-tiktoken) client.
    assert len(root_prompts[0]) < 40_000

    # The synthesis call actually ran and produced a real abstract — not the
    # empty string the unbudgeted version silently fell back to.
    assert summary == "tóm tắt tổng hợp"
