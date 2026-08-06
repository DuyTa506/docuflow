"""§3 picks disciplines by catalog code; strong research groups do not.

Two changes are pinned here:

* The catalog is the filtered Phụ lục I of Thông tư 09/2022, so a valid answer is
  a **7-digit discipline code**. A name does not identify a discipline — in the
  national catalog one name sits under several codes — so filtering by name, as
  before, is structurally wrong.

* `strong_research_groups` used to have to match a list of 18 groups that the
  catalog file itself admitted had no official names. Forcing the model to pick
  verbatim from an invented list produces invented output. The field is now free
  text from the model's own knowledge — in exchange it has **no guard against
  invention at all**, so only the cleaning can be pinned here: trim whitespace,
  drop empties, de-duplicate.

The catalog remains optional data: without it, skip the LLM call and say so.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

CATALOG = {
    "undergraduate": [
        {
            "code": "74801",
            "name": "Máy tính",
            "children": [
                {"code": "7480101", "name": "Khoa học máy tính"},
                {"code": "7480103", "name": "Kỹ thuật phần mềm"},
            ],
        }
    ],
    "master": [
        {
            "code": "85202",
            "name": "Kỹ thuật điện, điện tử và viễn thông",
            "children": [{"code": "8520204", "name": "Kỹ thuật rađa - dẫn đường"}],
        }
    ],
    "phd": [],
}


def _make_llm(payload):
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value="{}")
    llm.extract_json = MagicMock(return_value=payload)
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    return llm


def _session():
    s = MagicMock()
    s.__enter__ = MagicMock(return_value=s)
    s.__exit__ = MagicMock(return_value=False)
    s.query.return_value.filter.return_value.first.return_value = MagicMock()
    return s


async def _run(catalog, payload):
    from services.usage_scope_service import UsageScopeService

    svc = UsageScopeService()
    llm = _make_llm(payload)
    with (
        patch("services.usage_scope_service.get_db_manager") as dbm,
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch("utils.tree_payload.load_latest_tree_payload", return_value=None),
        patch.object(svc, "_read_text", return_value="nội dung tài liệu về máy tính"),
        patch.object(svc, "_progress"),
        patch("services.usage_scope_service.load_catalog", return_value=catalog),
    ):
        dbm.return_value.session.return_value = _session()
        result = await svc._extract("DOC_001", None)
    return result, llm


class TestCodeMatching:
    @pytest.mark.asyncio
    async def test_codes_resolve_to_official_names(self):
        result, _ = await _run(CATALOG, {"undergraduate": ["7480101"], "master": ["8520204"]})

        assert result["undergraduate"] == ["Khoa học máy tính"]
        assert result["master"] == ["Kỹ thuật rađa - dẫn đường"]

    @pytest.mark.asyncio
    async def test_name_answer_still_resolves(self):
        """The model does not always return the right shape."""
        result, _ = await _run(CATALOG, {"undergraduate": ["Ngành Khoa học máy tính"]})

        assert result["undergraduate"] == ["Khoa học máy tính"]

    @pytest.mark.asyncio
    async def test_invented_code_is_rejected(self):
        result, _ = await _run(CATALOG, {"undergraduate": ["7720101"]})

        assert result["undergraduate"] == []

    @pytest.mark.asyncio
    async def test_dropped_items_are_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="services.usage_scope_service"):
            await _run(CATALOG, {"undergraduate": ["7720101"]})

        assert "7720101" in caplog.text


class TestStrongResearchGroups:
    @pytest.mark.asyncio
    async def test_model_knowledge_is_kept_not_filtered(self):
        """No pick-list any more — the model's answer is kept as given."""
        result, _ = await _run(
            CATALOG,
            {"strong_research_groups": ["Trí tuệ nhân tạo", "An toàn thông tin"]},
        )

        assert result["strong_research_groups"] == ["Trí tuệ nhân tạo", "An toàn thông tin"]

    @pytest.mark.asyncio
    async def test_blanks_and_duplicates_are_cleaned(self):
        result, _ = await _run(
            CATALOG,
            {"strong_research_groups": ["  Trí tuệ nhân tạo  ", "", "Trí tuệ nhân tạo", None, 7]},
        )

        assert result["strong_research_groups"] == ["Trí tuệ nhân tạo"]


class TestPrompt:
    @pytest.mark.asyncio
    async def test_prompt_carries_the_tree_and_asks_for_codes(self):
        _, llm = await _run(CATALOG, {})

        prompt = llm.chat_completion.await_args.args[0]
        assert "74801 Máy tính" in prompt, "nhóm ngành cho mô hình định hướng"
        assert "7480101 Khoa học máy tính" in prompt
        assert "8520204" in prompt

    @pytest.mark.asyncio
    async def test_empty_level_is_not_offered(self):
        """When phd is empty, do not invite the model to pick at that level."""
        _, llm = await _run(CATALOG, {})

        prompt = llm.chat_completion.await_args.args[0]
        assert prompt.count("TIẾN SĨ") == 0


class TestOptionalCatalog:
    @pytest.mark.asyncio
    async def test_no_catalog_means_no_llm_call(self):
        """Asking the model to pick from an empty list just burns tokens."""
        result, llm = await _run({"undergraduate": [], "master": []}, {})

        llm.chat_completion.assert_not_awaited()
        assert result == {
            "undergraduate": [],
            "master": [],
            "phd": [],
            "strong_research_groups": [],
        }

    @pytest.mark.asyncio
    async def test_no_catalog_is_reported_not_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger="services.usage_scope_service"):
            await _run({}, {})

        assert "no ctđt catalog loaded" in caplog.text.casefold()

    @pytest.mark.asyncio
    async def test_partial_catalog_still_runs(self):
        """One level with disciplines is enough to be worth asking."""
        _, llm = await _run({"phd": CATALOG["master"]}, {})

        llm.chat_completion.assert_awaited_once()
