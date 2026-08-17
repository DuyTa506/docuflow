"""Tree quality gate tests."""

from utils.tree_quality import TREE_SCHEMA_VERSION, validate_tree_payload


def test_good_tree_passes():
    tree = {
        "title": "Document",
        "content": "Body text here.",
        "children": [
            {"title": "1. Introduction", "content": "Intro body.", "children": []},
            {"title": "2. Methods", "content": "Methods body.", "children": []},
        ],
    }
    quality = validate_tree_payload(tree, page_count=10)
    assert quality["ok"] is True
    assert quality["schema_version"] == TREE_SCHEMA_VERSION


def test_paragraph_titles_fail():
    paragraph = "x" * 200
    tree = {
        "title": "Document",
        "content": "",
        "children": [
            {"title": paragraph, "content": paragraph, "children": []},
            {"title": "Another " + paragraph, "content": paragraph, "children": []},
        ],
    }
    quality = validate_tree_payload(tree, page_count=5)
    assert quality["ok"] is False
