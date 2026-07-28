"""merge_nodes_content dropped the hierarchy level of merged paragraphs.

Phase 6 of build_spatial_tree assigns `final_level`; phase 6.5 (thinning) then
rebuilt merged nodes from scratch without it. build_tree_from_elements falls
back to `elem.get("final_level", elem.get("spatial_level", 3))`, so every merged
body paragraph re-entered the tree at level 3 *with its own body text as the
title* — manufacturing hundreds of pseudo-sections.

Observed on N4.11.160, where §2.2 of the digest listed a raw formula
`F = (( JAMZ И Z ) ИЛИ ( JAMN И N )) ИЛИ NEXT_ADDRESS [8]` as a "chapter".
"""

from core.spatial.spatial_tree_builder import build_tree_from_elements
from core.spatial.thinning import merge_nodes_content


def _text_node(y, text, *, final_level=5, page=1):
    return {
        "label": "text",
        "text_content": text,
        "text_full": text,
        "bbox_x1": 10,
        "bbox_y1": y,
        "bbox_x2": 300,
        "bbox_y2": y + 18,
        "page_number": page,
        "final_level": final_level,
        "spatial_level": final_level,
    }


def test_merged_node_keeps_constituent_level():
    merged = merge_nodes_content([_text_node(100, "first line"), _text_node(120, "second line")])

    assert merged["final_level"] == 5
    assert merged["spatial_level"] == 5


def test_merged_node_level_is_max_of_constituents():
    merged = merge_nodes_content(
        [_text_node(100, "a", final_level=4), _text_node(120, "b", final_level=5)]
    )

    assert merged["final_level"] == 5, "merged body must not float above its deepest part"


def test_merged_node_falls_back_to_spatial_level():
    nodes = [_text_node(100, "a"), _text_node(120, "b")]
    for n in nodes:
        n.pop("final_level")

    assert merge_nodes_content(nodes)["final_level"] == 5


def test_merged_paragraph_does_not_become_a_root_child():
    """The digest treats root children as chapters — body text must not land there."""
    heading = {
        "label": "title",
        "text_content": "Глава 1. Введение",
        "text_full": "Глава 1. Введение",
        "bbox_x1": 10,
        "bbox_y1": 40,
        "bbox_x2": 300,
        "bbox_y2": 60,
        "page_number": 1,
        "final_level": 0,
    }
    merged = merge_nodes_content([_text_node(100, "body one"), _text_node(120, "body two")])

    tree = build_tree_from_elements([heading, merged]).to_dict()

    root_titles = [c["title"] for c in tree["children"]]
    assert root_titles == ["Глава 1. Введение"], f"body text leaked into chapters: {root_titles}"


def test_merge_preserves_bbox_label_and_provenance():
    merged = merge_nodes_content([_text_node(100, "first"), _text_node(120, "second")])

    assert merged["label"] == "paragraph"
    assert (merged["bbox_x1"], merged["bbox_y1"]) == (10, 100)
    assert merged["bbox_y2"] == 138
    assert merged["text_content"] == "first second"
    assert merged["text_full"] == "first\nsecond"
    assert merged["merged_from"] == 2
    assert merged["original_labels"] == ["text", "text"]
    assert merged["page_number"] == 1
