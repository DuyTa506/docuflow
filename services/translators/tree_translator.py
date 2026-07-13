"""Tree-based translation fallback using TreeIndex structure."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from core.pageindex.enrichment.translator import StructuredTranslator

ProgressCallback = Optional[Callable[[int, str], Any]]


def _normalize_tree_roots(tree_data: dict) -> List[dict]:
    """Return a list of root nodes regardless of tree_data wrapper shape."""
    if not tree_data:
        return []
    if isinstance(tree_data, list):
        return tree_data
    children = tree_data.get("children") or tree_data.get("child_nodes") or []
    if children:
        return children
    if tree_data.get("title") or tree_data.get("content") or tree_data.get("text"):
        return [tree_data]
    return []


def _adapt_node_for_translator(node: dict) -> dict:
    """Map spatial tree node keys to StructuredTranslator expectations."""
    adapted = dict(node)
    if "content" in adapted and "text" not in adapted:
        adapted["text"] = adapted["content"]
    children = adapted.get("children") or adapted.get("child_nodes") or []
    if children:
        adapted["nodes"] = [_adapt_node_for_translator(c) for c in children]
    return adapted


def _flatten_translated_tree(nodes: List[dict]) -> str:
    parts: List[str] = []

    def walk(node: dict):
        title = (node.get("title") or "").strip()
        if title:
            parts.append(title)
        text = (node.get("text") or node.get("content") or node.get("text_content") or "").strip()
        if text:
            parts.append(text)
        for child in node.get("children") or node.get("child_nodes") or node.get("nodes") or []:
            walk(child)

    for root in nodes:
        walk(root)
    return "\n\n".join(parts)


class TreeTranslator:
    """Translate via TreeIndex hierarchy when layout elements are unavailable."""

    def __init__(self, translator: StructuredTranslator):
        self.translator = translator

    async def translate_tree(
        self,
        tree_data: dict,
        *,
        on_progress: ProgressCallback = None,
    ) -> dict:
        roots = _normalize_tree_roots(tree_data)
        adapted = [_adapt_node_for_translator(r) for r in roots]
        translated_structure = await self.translator.translate_structure(
            adapted, on_progress=on_progress
        )

        return {
            "translation_mode": "tree",
            "translated_elements": None,
            "translated_content": _flatten_translated_tree(translated_structure),
            "translated_file_path": None,
        }
