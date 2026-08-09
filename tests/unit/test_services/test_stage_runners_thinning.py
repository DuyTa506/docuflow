"""Regression test: the digest pipeline's BUILD_TREE stage must enable tree
thinning, otherwise large documents produce thousands of over-fragmented
nodes (confirmed on DOC_059: 5006 nodes, 252 raw-heading "chapters")."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_run_build_tree_enables_thinning():
    from services.pipeline.stage_runners import run_build_tree

    mock_svc = MagicMock()
    mock_svc.build_enhanced_tree_index = AsyncMock(return_value={"node_count": 1})

    with (
        patch("services.pipeline.stage_runners.get_db_manager") as mock_dbm,
        patch("services.pipeline.stage_runners.TreeIndexingService", return_value=mock_svc),
    ):
        mock_dbm.return_value.session.return_value.__enter__.return_value = MagicMock()
        mock_dbm.return_value.session.return_value.__exit__.return_value = False

        await run_build_tree("DOC_TEST")

    mock_svc.build_enhanced_tree_index.assert_awaited_once()
    _, kwargs = mock_svc.build_enhanced_tree_index.await_args
    assert kwargs["if_thinning"] is True
