"""Budget-aware document sampling for pipeline LLM stages.

Head-only truncation (`truncate_to_tokens(text, budget)`) blinds a stage to
everything past the first ~budget tokens — on a 700-page book that is >95%
of the document. `build_stratified_sample` instead spends the same budget on
full coverage: the chapter outline plus a proportional excerpt from every
chapter (tree available), or evenly spaced windows over the flat text.
"""

from __future__ import annotations

from typing import Callable, Optional

_OUTLINE_RATIO = 0.1
_OUTLINE_MAX_DEPTH = 3
_MIN_SECTION_TOKENS = 150
_WINDOW_TOKENS = 800
_CHARS_PER_TOKEN = 4  # sizing heuristic only; exact cuts go through `truncate`


def gather_node_text(node: dict, max_chars: int = 4000, max_nodes: int = 300) -> str:
    """Concatenate a node's own text with descendant text in document
    reading order (depth-first pre-order) until max_chars is reached.

    Handles heading-only nodes whose real body content lives one or more
    levels down in children. `max_nodes` bounds traversal independent of
    `max_chars` (a subtree of mostly empty nodes never adds to the char
    total, so the char budget alone can't stop the walk).
    """

    def _own_text(n: dict) -> str:
        return (
            n.get("content") or n.get("text") or n.get("text_content") or n.get("summary") or ""
        ).strip()

    parts: list[str] = []
    total_len = 0
    visited = 0
    stack = [node]

    while stack and total_len < max_chars and visited < max_nodes:
        current = stack.pop()
        visited += 1
        text = _own_text(current)
        if text:
            parts.append(text)
            total_len += len(text)
        children = current.get("children") or current.get("child_nodes") or []
        for child in reversed(children):  # reversed + LIFO pop = left-to-right pre-order
            stack.append(child)

    return "\n\n".join(parts)


def _build_outline(tree_data: dict, token_budget: int, count_tokens: Callable) -> str:
    lines: list[str] = []
    used = 0

    def walk(node: dict, depth: int) -> None:
        nonlocal used
        if depth > _OUTLINE_MAX_DEPTH or used >= token_budget:
            return
        title = (node.get("title") or "").strip()
        if title and depth > 0:  # skip the root/document title line
            line = "  " * (depth - 1) + f"- {title}"
            cost = count_tokens(line)
            if used + cost > token_budget:
                return
            lines.append(line)
            used += cost
        for child in node.get("children") or node.get("child_nodes") or []:
            walk(child, depth + 1)

    walk(tree_data, 0)
    return "\n".join(lines)


def _window_sample(
    flat_text: str,
    token_budget: int,
    count_tokens: Callable,
    truncate: Callable,
) -> str:
    if count_tokens(flat_text) <= token_budget:
        return flat_text
    k = max(2, token_budget // _WINDOW_TOKENS)
    # Reserve separator overhead inside the per-window size — otherwise the
    # final truncate clips the tail window (i.e. the document's ending).
    per_window_tokens = max(50, token_budget // k - 8)
    window_chars = per_window_tokens * _CHARS_PER_TOKEN
    n = len(flat_text)
    span = max(0, n - window_chars)
    # Evenly spaced; first window starts at 0, last is anchored to the exact
    # end of the text (integer rounding must never clip the document tail).
    positions = [round(i * span / (k - 1)) for i in range(k)]
    parts = [flat_text[p : p + window_chars] for p in positions]
    return truncate("\n\n[...]\n\n".join(parts), token_budget)


def build_stratified_sample(
    tree_data: Optional[dict],
    flat_text: str,
    *,
    token_budget: int,
    count_tokens: Callable,
    truncate: Callable,
    min_section_tokens: int = _MIN_SECTION_TOKENS,
) -> str:
    """Build a whole-document sample within `token_budget` tokens.

    Tree path: chapter/section title outline (+~10% of budget) followed by a
    per-chapter excerpt, budget allocated proportionally to chapter mass with
    a floor so small chapters are never squeezed out.
    No tree: evenly spaced windows over the flat text (head always included).
    """
    chapters = []
    if tree_data:
        chapters = [
            ch
            for ch in (tree_data.get("children") or tree_data.get("child_nodes") or [])
            if isinstance(ch, dict)
        ]

    if not chapters:
        return _window_sample(flat_text or "", token_budget, count_tokens, truncate)

    outline = _build_outline(tree_data, max(1, int(token_budget * _OUTLINE_RATIO)), count_tokens)
    remaining = max(min_section_tokens, token_budget - count_tokens(outline))

    gathered = [gather_node_text(ch, max_chars=remaining * _CHARS_PER_TOKEN) for ch in chapters]
    entries = [
        ((ch.get("title") or f"Phần {i + 1}").strip(), text)
        for i, (ch, text) in enumerate(zip(chapters, gathered))
        if text.strip()
    ]
    if not entries:
        return _window_sample(flat_text or "", token_budget, count_tokens, truncate)

    masses = [max(1, count_tokens(text)) for _, text in entries]
    total_mass = sum(masses)
    header_overhead = 10 * len(entries)
    text_budget = max(min_section_tokens * len(entries), remaining - header_overhead)

    allocs = [max(min_section_tokens, int(text_budget * mass / total_mass)) for mass in masses]
    overshoot = sum(allocs)
    if overshoot > text_budget:
        scale = text_budget / overshoot
        allocs = [max(min_section_tokens, int(a * scale)) for a in allocs]

    blocks = [f"OUTLINE:\n{outline}"] if outline else []
    for i, ((title, text), alloc) in enumerate(zip(entries, allocs), start=1):
        blocks.append(f"## [{i}] {title}\n{truncate(text, alloc)}")

    return "\n\n".join(blocks)


def build_pipeline_doc_sample(document_id: str, text: str, enricher, token_budget: int) -> str:
    """Stage-facing helper: latest tree payload (if any) + stratified sample."""
    from utils.tree_payload import load_latest_tree_payload

    tree_data = load_latest_tree_payload(document_id)
    return build_stratified_sample(
        tree_data,
        text,
        token_budget=token_budget,
        count_tokens=enricher.count_tokens,
        truncate=enricher.truncate_to_tokens,
    )
