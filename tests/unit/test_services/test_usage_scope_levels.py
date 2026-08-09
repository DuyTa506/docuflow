"""§3 has to decide each training level on its own terms.

The prompt asked one question — "which disciplines could use this document?" —
and printed the answer into three keys. Nothing in it said what makes a level
different, so the three lists came out as noise around a single answer: over
three live runs on DOC_002 the doctoral list was always a subset of the other
two, nothing was ever doctoral-only, and the undergraduate count swung from 4
to 11 between runs.

They cannot all be right, because the catalog itself is not the same at every
level: 147 disciplines at ĐH, 141 at ThS, 133 at TS, and 54 of them exist only
at ĐH. `resolve_items` already drops a code offered at a level where it does
not exist — twice in those runs — so a copied list is not merely imprecise, it
wastes picks.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.usage_scope_service import UsageScopeService

CATALOG = {
    "undergraduate": [
        {"code": "74802", "name": "Máy tính", "children": [{"code": "7480201", "name": "CNTT"}]}
    ],
    "master": [
        {"code": "84802", "name": "Máy tính", "children": [{"code": "8480201", "name": "CNTT"}]}
    ],
    "phd": [
        {"code": "94802", "name": "Máy tính", "children": [{"code": "9480201", "name": "CNTT"}]}
    ],
}


@pytest.fixture
def prompt():
    """The prompt §3 actually sends."""
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="{}")
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.extract_json = MagicMock(return_value={})
    llm.encoding = None

    with (
        patch("services.usage_scope_service.load_catalog", return_value=CATALOG),
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch.object(UsageScopeService, "_read_text", return_value="Nội dung tài liệu. " * 200),
        patch("utils.doc_sampling.build_pipeline_doc_sample", side_effect=lambda d, t, e, b: t),
        patch.object(UsageScopeService, "_save"),
        patch.object(UsageScopeService, "_progress"),
    ):
        import asyncio

        asyncio.run(UsageScopeService()._extract("DOC_001"))

    return llm.chat_completion.await_args.args[0]


class TestLevelsAreDistinguished:
    @pytest.mark.parametrize(
        "level,marker",
        [
            ("undergraduate", "foundation"),
            ("master", "specialis"),
            ("phd", "research"),
        ],
    )
    def test_each_level_gets_its_own_criterion(self, prompt, level, marker):
        """Naming the level is not enough — it needs the question to ask there."""
        assert marker in prompt.lower(), f"no criterion for {level}"

    def test_the_three_lists_are_decided_separately(self, prompt):
        low = prompt.lower()
        assert "separately" in low or "independent" in low

    def test_copying_one_list_into_all_three_is_called_out(self, prompt):
        """The observed failure is the one the prompt must name explicitly."""
        assert "same list" in prompt.lower()

    def test_the_rule_is_repeated_at_the_generation_point(self, prompt):
        """Stating a critical rule twice is what made NUMERIC_FIDELITY hold in
        main_content; the reminder belongs after the excerpt, next to 'JSON:'."""
        tail = prompt[prompt.rindex("DOCUMENT EXCERPT:") :].lower()
        assert "level" in tail and ("separate" in tail or "independent" in tail)


def test_a_level_with_no_disciplines_gets_no_criterion():
    """The criteria must come from the same list the catalog blocks do.

    An empty section in the prompt is an invitation to invent something to fill
    it — which is why `test_empty_level_is_not_offered` exists. Spelling the
    levels out by hand in the rules re-introduced exactly that, for a level the
    catalog cannot answer.
    """
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value="{}")
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.extract_json = MagicMock(return_value={})
    llm.encoding = None

    no_phd = {**CATALOG, "phd": []}
    with (
        patch("services.usage_scope_service.load_catalog", return_value=no_phd),
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch.object(UsageScopeService, "_read_text", return_value="Nội dung. " * 200),
        patch("utils.doc_sampling.build_pipeline_doc_sample", side_effect=lambda d, t, e, b: t),
        patch.object(UsageScopeService, "_save"),
        patch.object(UsageScopeService, "_progress"),
    ):
        import asyncio

        asyncio.run(UsageScopeService()._extract("DOC_001"))

    prompt = llm.chat_completion.await_args.args[0]
    assert "TIẾN SĨ" not in prompt
    assert "ORIGINAL research" not in prompt
    # The levels that do exist keep theirs.
    assert "foundation material" in prompt and "specialise" in prompt


def test_the_catalog_differing_per_level_is_stated(prompt):
    """A code valid at one level may not exist at another — the model should be
    told, rather than spending picks that `resolve_items` then drops."""
    assert "not the same at every level" in prompt.lower()
