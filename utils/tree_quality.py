"""Post-build validation for TreeIndex payloads."""

from __future__ import annotations

from typing import Any

from utils.structural_titles import is_structural_title

TREE_SCHEMA_VERSION = 1
_MAX_INVALID_TITLE_RATIO = 0.35
_MIN_STRUCTURAL_NODES = 1


def _walk(node: dict, page_numbers: set[int]) -> tuple[int, int, int, int]:
    """Return structural, invalid-title, total-with-title, content-char counts."""
    title = (node.get("title") or node.get("name") or "").strip()
    label = node.get("label") or node.get("node_type")
    body = (
        node.get("content")
        or node.get("text")
        or node.get("text_content")
        or node.get("text_full")
        or ""
    )
    structural = 0
    invalid = 0
    titled = 0
    content_chars = len(str(body))

    if title and title.lower() != "document":
        titled += 1
        if is_structural_title(title, label=label, body=body):
            structural += 1
        else:
            invalid += 1

    page = node.get("page_number") or node.get("page_start")
    if isinstance(page, int) and page > 0:
        page_numbers.add(page)

    children = node.get("children") or node.get("child_nodes") or node.get("nodes") or []
    for child in children:
        if isinstance(child, dict):
            s, i, t, c = _walk(child, page_numbers)
            structural += s
            invalid += i
            titled += t
            content_chars += c

    return structural, invalid, titled, content_chars


def validate_tree_payload(tree: dict | None, *, page_count: int = 1) -> dict[str, Any]:
    if not tree or not isinstance(tree, dict):
        return {
            "ok": False,
            "schema_version": TREE_SCHEMA_VERSION,
            "reason": "empty_tree",
        }

    page_numbers: set[int] = set()
    structural, invalid, titled, content_chars = _walk(tree, page_numbers)
    invalid_ratio = (invalid / titled) if titled else 1.0
    ok = (
        structural >= _MIN_STRUCTURAL_NODES
        and invalid_ratio <= _MAX_INVALID_TITLE_RATIO
        and content_chars > 0
    )

    return {
        "ok": ok,
        "schema_version": TREE_SCHEMA_VERSION,
        "structural_nodes": structural,
        "invalid_title_nodes": invalid,
        "titled_nodes": titled,
        "invalid_title_ratio": round(invalid_ratio, 4),
        "content_chars": content_chars,
        "page_span": len(page_numbers) or max(1, int(page_count or 1)),
        "max_depth": _max_depth(tree),
    }


def _max_depth(node: dict, depth: int = 0) -> int:
    children = node.get("children") or node.get("child_nodes") or node.get("nodes") or []
    if not children:
        return depth
    return max(_max_depth(child, depth + 1) for child in children if isinstance(child, dict))
