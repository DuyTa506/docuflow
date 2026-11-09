"""Rebuild nested tree dicts from flat TreeNode rows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_tree_dict(
    nodes: List[Any],
    *,
    root_node_id: Optional[str] = None,
) -> Dict:
    """
    Reconstruct a nested tree dict from flat TreeNode ORM rows.

    Falls back to the first root (parent_node_id is None) when root_node_id
    is not provided.
    """
    if not nodes:
        return {}

    by_id: Dict[str, Dict] = {}
    children_map: Dict[str, List[str]] = {}

    for node in nodes:
        node_id = getattr(node, "node_id", None) or ""
        if not node_id:
            continue
        entry: Dict[str, Any] = {
            "node_id": node_id,
            "id": node_id,
            "type": getattr(node, "node_type", None),
            "node_type": getattr(node, "node_type", None),
            "title": getattr(node, "title", None),
            "name": getattr(node, "title", None),
            "summary": getattr(node, "summary", None),
            "node_summary": getattr(node, "summary", None),
            "page_start": getattr(node, "page_start", None),
            "start_page": getattr(node, "page_start", None),
            "page_end": getattr(node, "page_end", None),
            "end_page": getattr(node, "page_end", None),
            "token_count": getattr(node, "token_count", None),
            "tokens": getattr(node, "token_count", None),
            "children": [],
            "child_nodes": [],
        }
        by_id[node_id] = entry
        parent = getattr(node, "parent_node_id", None)
        if parent:
            children_map.setdefault(parent, []).append(node_id)

    for parent_id, child_ids in children_map.items():
        parent = by_id.get(parent_id)
        if not parent:
            continue
        children = [by_id[cid] for cid in child_ids if cid in by_id]
        parent["children"] = children
        parent["child_nodes"] = children

    if root_node_id and root_node_id in by_id:
        return by_id[root_node_id]

    child_ids = {cid for ids in children_map.values() for cid in ids}
    roots = [by_id[nid] for nid in by_id if nid not in child_ids]
    if len(roots) == 1:
        return roots[0]
    if roots:
        return {"node_id": "root", "children": roots, "child_nodes": roots}
    return by_id.get(next(iter(by_id)), {})
