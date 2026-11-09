"""Load tree index payloads from DB JSON or MinIO."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from core.pageindex.tree_builder import build_tree_dict
from data.db_models import TreeIndex, TreeNode
from services.object_storage import get_object_storage


def get_tree_payload(
    db: Session,
    tree_index: TreeIndex,
    *,
    nodes: Optional[List[TreeNode]] = None,
) -> Dict[str, Any]:
    """
    Return the full nested tree dict for a TreeIndex.

    Resolution order:
    1. tree_data JSON column (legacy / primary during transition)
    2. tree_data_key in MinIO
    3. rebuild from tree_nodes (best-effort)
    """
    if tree_index.tree_data:
        return tree_index.tree_data

    if tree_index.tree_data_key:
        storage = get_object_storage()
        if storage.exists(tree_index.tree_data_key):
            raw = storage.get_bytes(tree_index.tree_data_key)
            return json.loads(raw.decode("utf-8"))

    node_rows = nodes
    if node_rows is None:
        node_rows = db.query(TreeNode).filter(TreeNode.tree_index_id == tree_index.id).all()
    rebuilt = build_tree_dict(node_rows)
    return rebuilt or {}


def load_latest_tree_payload(document_id: str) -> Optional[Dict[str, Any]]:
    """Load the newest TreeIndex payload for a document, or None if absent.

    Opens its own session — for callers (pipeline stages) that don't already
    hold one where the tree is needed.
    """
    from data.database import get_db_manager

    with get_db_manager().session() as db:
        tree_index = (
            db.query(TreeIndex)
            .filter(TreeIndex.document_id == document_id)
            .order_by(TreeIndex.created_at.desc())
            .first()
        )
        if not tree_index:
            return None
        return get_tree_payload(db, tree_index)
