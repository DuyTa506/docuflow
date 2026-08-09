"""Tree-mode translation shipped the *untranslated* tree.

Confirmed on N4.11.160 (816-page Russian book, tree mode): 97.1% of the
exported pages were still Russian and 39.3% of paragraph blocks were exact
immediate duplicates, blowing 816 source pages up to 2238.

Two independent defects, both here:

1. ``_adapt_node_for_translator`` copies the node with ``dict(node)`` — keeping
   the raw ``children`` — and *adds* an adapted ``nodes`` list.
   ``StructuredTranslator.translate_structure`` only recurses through ``nodes``
   (translator.py), so ``children`` stays in the source language. The flatten
   walk then read ``children`` first, i.e. every non-root node was emitted
   untranslated. Only root-level nodes ever reached the output in Vietnamese.

2. ``tree_indexing_service`` seeds ``text_full`` from ``text_content``, so a
   node's ``title`` and ``content`` are the same string. Both were translated —
   by two different prompts — and both were emitted, adjacent. Hence the
   duplicate blocks, and two diverging renderings of one sentence.
"""

from unittest.mock import MagicMock, patch

import pytest

from core.pageindex.enrichment.translator import StructuredTranslator
from services.translators.tree_translator import (
    TreeTranslator,
    _adapt_node_for_translator,
    _flatten_translated_tree,
)


def _tree():
    """Spatial-tree shape: nested ``children``, body text under ``content``."""
    return {
        "title": "Document",
        "children": [
            {
                "title": "Глава 1. Введение",
                "content": "Тело главы один.",
                "children": [
                    {
                        "title": "Многоуровневая организация",
                        "content": "Тело раздела 1.1.",
                        "children": [],
                    }
                ],
            }
        ],
    }


async def _fake_translate(text):
    return f"VI:{text}"


def _translator():
    return StructuredTranslator(llm_client=MagicMock(), source_lang="ru", target_lang="vi")


@pytest.mark.asyncio
async def test_flatten_uses_translated_nodes_not_raw_children():
    """Every node — not just the roots — must reach the output translated."""
    translator = _translator()

    with (
        patch.object(translator, "translate_title", side_effect=_fake_translate),
        patch.object(translator, "translate_text", side_effect=_fake_translate),
        patch.object(translator, "translate_text_chunked", side_effect=_fake_translate),
    ):
        result = await TreeTranslator(translator).translate_tree(_tree())

    content = result["translated_content"]
    assert "VI:Глава 1. Введение" in content
    assert "VI:Многоуровневая организация" in content, "descendant emitted untranslated"
    assert "VI:Тело раздела 1.1." in content, "descendant body emitted untranslated"
    # nothing may survive in the source language
    for raw in ("Глава 1. Введение", "Многоуровневая организация", "Тело раздела 1.1."):
        assert f"\n{raw}\n" not in f"\n{content}\n"


def test_adapt_drops_raw_children_key():
    """The raw subtree must not travel alongside the adapted one.

    Keeping both is what let the flatten walk pick the untranslated branch, and
    it doubles the memory of translate_structure's deepcopy on a book-sized tree.
    """
    adapted = _adapt_node_for_translator(_tree()["children"][0])

    assert "children" not in adapted
    assert "child_nodes" not in adapted
    assert [n["title"] for n in adapted["nodes"]] == ["Многоуровневая организация"]
    assert "children" not in adapted["nodes"][0]


@pytest.mark.asyncio
async def test_identical_title_and_content_translated_once():
    """title == content (the text_full aliasing) must cost one call, not two."""
    translator = _translator()
    tree = {
        "title": "Document",
        "children": [{"title": "Сборка модулей памяти", "content": "Сборка модулей памяти"}],
    }

    with (
        patch.object(translator, "translate_title", side_effect=_fake_translate) as title_mock,
        patch.object(translator, "translate_text", side_effect=_fake_translate),
        patch.object(
            translator, "translate_text_chunked", side_effect=_fake_translate
        ) as text_mock,
    ):
        result = await TreeTranslator(translator).translate_tree(tree)

    assert title_mock.await_count + text_mock.await_count == 1
    assert result["translated_content"].count("VI:Сборка модулей памяти") == 1


def test_flatten_emits_no_duplicate_block():
    """Two adjacent identical blocks is the signature of the 2238-page export."""
    nodes = [{"title": "Сборка модулей", "text": "Сборка модулей", "nodes": []}]

    parts = _flatten_translated_tree(nodes).split("\n\n")

    assert parts == ["Сборка модулей"]


def test_long_body_as_title_keeps_text_field():
    """Thinning artifacts carry a whole paragraph as the title.

    Those must be translated as body prose, not through the "keep it concise"
    title prompt — so the surviving field is ``text``, and the title is cleared
    rather than emitted twice.
    """
    body = (
        "ПРИ ОФОРМЛЕНИИ ЗАКАЗА УКАЖИТЕ адрес доставки, полное имя получателя "
        "и контактный телефон, а также выберите удобный способ оплаты заказа "
        "и подтвердите согласие на обработку персональных данных перед отправкой."
    )
    assert len(body) > 150, "fixture must exercise the long-body branch"
    adapted = _adapt_node_for_translator({"title": body, "content": body})

    assert adapted.get("title") in (None, "")
    assert adapted["text"] == body
