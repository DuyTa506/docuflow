"""§3 hướng nghiên cứu was empty and reported COMPLETED.

Three failures stacked: the call passed no `max_tokens` so a 20-item JSON
answer got cut mid-object; `_extract_json` turned the resulting parse error
into `[]`; and the store step deleted the previous run's associations *before*
discovering it had nothing to put back. A truncated response therefore
destroyed a good earlier result and still looked like success.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import settings

VALID = '[{"direction_name": "Kiến trúc máy tính", "is_predefined": false, "confidence": 0.9}]'
TRUNCATED = '[{"direction_name": "Kiến trúc máy tính", "is_pre'


def _make_llm(response):
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value=response)
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    llm.extract_json = MagicMock(side_effect=_fake_extract_json)
    return llm


def _fake_extract_json(response):
    import json

    return json.loads(response)


def _session(existing_assoc_count=0):
    s = MagicMock()
    s.__enter__ = MagicMock(return_value=s)
    s.__exit__ = MagicMock(return_value=False)
    filtered = s.query.return_value.filter.return_value
    filtered.all.return_value = []
    filtered.first.return_value = None
    filtered.count.return_value = existing_assoc_count
    filtered.order_by.return_value.first.return_value = None
    return s, filtered


async def _run(response, existing=0):
    from services.research_direction_service import ResearchDirectionService

    svc = ResearchDirectionService()
    llm = _make_llm(response)
    session, filtered = _session(existing)

    with (
        patch("services.research_direction_service.get_db_manager") as dbm,
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch.object(svc, "_read_text", return_value="nội dung tài liệu"),
        patch.object(svc, "_progress"),
    ):
        dbm.return_value.session.return_value = session
        result = await svc._do_extract("DOC_001", extraction_id=None)
    return result, llm, filtered


class TestOutputBudget:
    @pytest.mark.asyncio
    async def test_max_tokens_is_passed(self):
        """No cap at all is why the JSON came back cut in half."""
        _, llm, _ = await _run(VALID)

        assert llm.chat_completion.await_args.kwargs["max_tokens"] > 0

    @pytest.mark.asyncio
    async def test_requested_item_count_comes_from_settings(self):
        _, llm, _ = await _run(VALID)

        prompt = llm.chat_completion.await_args.args[0]
        assert f"up to {settings.research_directions_max_items}" in prompt


class TestParseFailure:
    @pytest.mark.asyncio
    async def test_truncated_json_raises_instead_of_reporting_success(self):
        with pytest.raises(ValueError, match="JSON"):
            await _run(TRUNCATED)

    @pytest.mark.asyncio
    async def test_truncated_json_does_not_delete_previous_directions(self):
        from services.research_direction_service import ResearchDirectionService

        svc = ResearchDirectionService()
        llm = _make_llm(TRUNCATED)
        session, filtered = _session()

        with (
            patch("services.research_direction_service.get_db_manager") as dbm,
            patch("api.dependencies.get_llm_client", return_value=llm),
            patch.object(svc, "_read_text", return_value="nội dung"),
            patch.object(svc, "_progress"),
            pytest.raises(ValueError),
        ):
            dbm.return_value.session.return_value = session
            await svc._do_extract("DOC_001", extraction_id=None)

        filtered.delete.assert_not_called()


class TestEmptyResult:
    @pytest.mark.asyncio
    async def test_valid_but_empty_keeps_previous_directions(self):
        """ "No directions this run" must not wipe a good earlier run."""
        result, _, filtered = await _run("[]", existing=4)

        filtered.delete.assert_not_called()
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_non_empty_result_still_replaces_previous(self):
        _, _, filtered = await _run(VALID)

        filtered.delete.assert_called_once()
