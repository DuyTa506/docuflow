"""Tree-based translation fallback using TreeIndex structure."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from core.pageindex.enrichment.translator import StructuredTranslator

ProgressCallback = Optional[Callable[[int, str], Any]]

# A node title longer than this is a thinning artifact carrying a whole
# paragraph, not a heading — translate it as body prose instead.
_TITLE_MAX_CHARS = 150


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
    """Map spatial tree node keys to StructuredTranslator expectations.

    The raw ``children``/``child_nodes`` keys are dropped: ``translate_structure``
    only recurses through ``nodes``, so leaving them behind means the untranslated
    subtree travels alongside the translated one — and whoever reads ``children``
    first gets the source language back.
    """
    adapted = dict(node)

    content = adapted.pop("content", None)
    if content and not adapted.get("text"):
        adapted["text"] = content

    title = (adapted.get("title") or "").strip()
    text = (adapted.get("text") or "").strip()
    if title and title == text:
        # Upstream seeds text_full from text_content, so a node's title and body
        # are routinely the same string. Translating both costs two LLM calls
        # and emits two diverging renderings of one sentence.
        if len(title) <= _TITLE_MAX_CHARS:
            adapted.pop("text", None)
        else:
            adapted["title"] = ""

    children = adapted.pop("children", None) or []
    child_nodes = adapted.pop("child_nodes", None) or []
    children = children or child_nodes
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
        if text and text != title:
            parts.append(text)
        # "nodes" holds the translated subtree; the raw keys are only a fallback
        # for callers that hand us an already-flat structure.
        for child in node.get("nodes") or node.get("children") or node.get("child_nodes") or []:
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
