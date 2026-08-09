"""§3 constrains decoding to the codes that exist, instead of asking nicely.

Every code §3 had rejected was clerical rather than a judgement error: a number
interpolated into a gap in the sequence (7520106, where the real one runs …103,
114, 115…), or a discipline moved to a level that does not offer it (7520203 at
undergraduate). Both are decidable before a token is emitted — the valid set is
known and finite.

Measured on DOC_002, ten runs each: 12 invalid codes out of 255 unconstrained,
0 out of 208 with a per-level `enum`, and 10% faster because the token space is
narrower. Told outright to emit an invented code the model could not.

What it does NOT fix is picking a valid code for the wrong reason. The
constraint knows which disciplines exist, not which ones fit.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.usage_scope_service import UsageScopeService
from utils.ctdt_catalog import response_schema

CATALOG = {
    "undergraduate": [
        {
            "code": "74802",
            "name": "Máy tính",
            "children": [
                {"code": "7480201", "name": "CNTT"},
                {"code": "7480108", "name": "Công nghệ kỹ thuật máy tính"},
            ],
        }
    ],
    "master": [
        {"code": "84802", "name": "Máy tính", "children": [{"code": "8480201", "name": "CNTT"}]}
    ],
    "phd": [
        {"code": "94802", "name": "Máy tính", "children": [{"code": "9480201", "name": "CNTT"}]}
    ],
}


class TestSchema:
    def test_each_level_is_limited_to_its_own_codes(self):
        props = response_schema(CATALOG)["properties"]

        assert props["undergraduate"]["items"]["enum"] == ["7480201", "7480108"]
        assert props["master"]["items"]["enum"] == ["8480201"]
        assert props["phd"]["items"]["enum"] == ["9480201"]

    def test_a_code_from_another_level_is_not_permitted(self):
        """The level-digit rewrite is what this closes: 8480108 does not exist."""
        enum = response_schema(CATALOG)["properties"]["master"]["items"]["enum"]

        assert "8480108" not in enum
        assert "7480108" not in enum

    def test_research_groups_stay_free_text(self):
        """The one field with no catalog cannot be constrained to one."""
        groups = response_schema(CATALOG)["properties"]["strong_research_groups"]

        assert groups["items"] == {"type": "string"}

    def test_a_level_with_no_disciplines_is_absent(self):
        """Same rule the prompt follows — an empty enum permits nothing at all,
        which would force the model to return an empty list for that level even
        when the catalog simply was not loaded for it."""
        schema = response_schema({**CATALOG, "phd": []})

        assert "phd" not in schema["properties"]
        assert "phd" not in schema["required"]

    def test_no_catalog_yields_no_level_constraints(self):
        schema = response_schema({"undergraduate": [], "master": [], "phd": []})

        assert set(schema["properties"]) == {"strong_research_groups"}


def _llm(payload='{"undergraduate": ["7480201"]}'):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=payload)
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.extract_json = MagicMock(return_value={"undergraduate": ["7480201"]})
    llm.encoding = None
    return llm


async def _run(llm):
    with (
        patch("services.usage_scope_service.load_catalog", return_value=CATALOG),
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch.object(UsageScopeService, "_read_text", return_value="Nội dung. " * 200),
        patch("utils.doc_sampling.build_pipeline_doc_sample", side_effect=lambda d, t, e, b: t),
        patch.object(UsageScopeService, "_save"),
        patch.object(UsageScopeService, "_progress"),
    ):
        return await UsageScopeService()._extract("DOC_001")


class TestTheServiceUsesIt:
    @pytest.mark.asyncio
    async def test_the_schema_is_sent_with_the_prompt(self):
        llm = _llm()

        await _run(llm)

        fmt = llm.chat_completion.await_args.kwargs["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["schema"] == response_schema(CATALOG)

    @pytest.mark.asyncio
    async def test_a_provider_that_rejects_the_schema_still_gets_an_answer(self):
        """Not every OpenAI-compatible server enforces `response_format`; some
        reject the request outright. Losing §3 over a parameter is worse than
        one retry — `resolve_items` still guards the unconstrained answer."""
        llm = _llm()
        llm.chat_completion = AsyncMock(
            side_effect=[ValueError("unknown parameter: response_format"), '{"x": 1}']
        )

        result = await _run(llm)

        assert llm.chat_completion.await_count == 2
        assert "response_format" not in llm.chat_completion.await_args.kwargs
        assert result["undergraduate"] == ["CNTT"]

    @pytest.mark.asyncio
    async def test_a_second_failure_is_not_swallowed(self):
        """Retrying once is a fallback; retrying forever hides an outage."""
        llm = _llm()
        llm.chat_completion = AsyncMock(side_effect=RuntimeError("provider down"))

        with pytest.raises(RuntimeError):
            await _run(llm)

        assert llm.chat_completion.await_count == 2
