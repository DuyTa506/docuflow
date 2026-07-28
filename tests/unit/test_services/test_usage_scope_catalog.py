"""§3 came back empty for N4.11.160 with no warning of any kind.

Two independent causes, both fixed here:

* the catalog filter compared raw strings, so an answer that differed by a
  dash or the "Ngành " prefix was thrown away silently;
* the catalog is optional operational data — when a deployment has none, the
  stage used to still spend an LLM call asking the model to pick from an
  empty list, then store an empty result indistinguishable from "no match".
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

CATALOG = {
    "undergraduate": ["Ngành Khoa học máy tính", "Ngành Kỹ thuật phần mềm"],
    "master": ["Ngành Kỹ thuật ra đa – dẫn đường"],
    "phd": [],
    "strong_research_groups": ["Trí tuệ nhân tạo và khoa học dữ liệu"],
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


class TestTolerantMatching:
    @pytest.mark.asyncio
    async def test_dash_and_prefix_near_miss_is_kept(self):
        """The exact-string filter is what emptied §3."""
        result, _ = await _run(
            CATALOG,
            {
                "undergraduate": ["Khoa học máy tính"],
                "master": ["Ngành Kỹ thuật ra đa - dẫn đường"],
            },
        )

        assert result["undergraduate"] == ["Ngành Khoa học máy tính"]
        assert result["master"] == ["Ngành Kỹ thuật ra đa – dẫn đường"]

    @pytest.mark.asyncio
    async def test_invented_programme_is_still_rejected(self):
        result, _ = await _run(CATALOG, {"undergraduate": ["Ngành Y khoa"]})

        assert result["undergraduate"] == []

    @pytest.mark.asyncio
    async def test_dropped_items_are_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="services.usage_scope_service"):
            await _run(CATALOG, {"undergraduate": ["Ngành Y khoa"]})

        assert "Ngành Y khoa" in caplog.text


class TestOptionalCatalog:
    @pytest.mark.asyncio
    async def test_no_catalog_means_no_llm_call(self):
        """Asking a model to choose from an empty list burns tokens for nothing."""
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

        assert "CTĐT" in caplog.text

    @pytest.mark.asyncio
    async def test_partial_catalog_still_runs(self):
        """One populated key is enough to be worth asking about."""
        _, llm = await _run({"phd": ["Ngành Toán ứng dụng"]}, {"phd": ["Ngành Toán ứng dụng"]})

        llm.chat_completion.assert_awaited_once()
