"""TreeTranslator.translate_tree() only reported progress once, at 95%, after
the whole tree finished translating — on documents routed through this path
(>2500 layout elements) the UI shows 0% for the entire run instead of ticking
up as nodes complete.
"""

from unittest.mock import AsyncMock

import pytest

from services.translators.tree_translator import TreeTranslator


@pytest.mark.asyncio
async def test_translate_tree_forwards_on_progress_to_translate_structure():
    translator = AsyncMock()
    translator.translate_structure = AsyncMock(return_value=[{"title": "T", "text": "body"}])
    tree_translator = TreeTranslator(translator)

    calls = []

    async def on_progress(pct, msg):
        calls.append((pct, msg))

    tree_data = {"children": [{"title": "T", "content": "body"}]}
    await tree_translator.translate_tree(tree_data, on_progress=on_progress)

    translator.translate_structure.assert_awaited_once()
    _, kwargs = translator.translate_structure.call_args
    assert kwargs.get("on_progress") is on_progress
