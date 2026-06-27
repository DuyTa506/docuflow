"""Tests for tree rebuild from flat nodes."""

from types import SimpleNamespace

from core.pageindex.tree_builder import build_tree_dict


def test_build_tree_dict_nested_children():
    nodes = [
        SimpleNamespace(
            node_id="root",
            node_type="section",
            title="Root",
            summary="sum",
            parent_node_id=None,
            page_start=1,
            page_end=2,
            token_count=10,
        ),
        SimpleNamespace(
            node_id="child",
            node_type="subsection",
            title="Child",
            summary=None,
            parent_node_id="root",
            page_start=1,
            page_end=1,
            token_count=5,
        ),
    ]
    tree = build_tree_dict(nodes)
    assert tree["title"] == "Root"
    assert len(tree["children"]) == 1
    assert tree["children"][0]["title"] == "Child"
