"""Hướng nghiên cứu is a proposal, not an extraction.

The stage used to hand the model a catalog and ask it to pick matches, so the
best it could do was re-label the document against 18 existing NNC groups.
A research direction has no ground-truth list to check against — the useful
answer comes from the model's own knowledge of the field, anchored only by
the document itself.

Membership in the catalog is therefore decided *by us* after the fact, not
claimed by the model: `is_predefined` is a lookup, not an opinion.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

CATALOG_NAMES = [
    "Trí tuệ nhân tạo và khoa học dữ liệu",
    "Kỹ thuật ra đa – dẫn đường",
]


def _make_llm(response):
    llm = AsyncMock()
    llm.chat_completion = AsyncMock(return_value=response)
    llm.count_tokens = MagicMock(side_effect=lambda t: max(1, len(t) // 4))
    llm.encoding = None
    llm.extract_json = MagicMock(side_effect=lambda r: __import__("json").loads(r))
    return llm


def _session(catalog_names):
    s = MagicMock()
    s.__enter__ = MagicMock(return_value=s)
    s.__exit__ = MagicMock(return_value=False)
    filtered = s.query.return_value.filter.return_value
    filtered.all.return_value = [
        MagicMock(direction_name=n, is_predefined=True) for n in catalog_names
    ]
    filtered.first.return_value = None  # every direction is a new row
    filtered.count.return_value = 0
    filtered.order_by.return_value.first.return_value = None
    return s


async def _run(response, catalog_names=CATALOG_NAMES):
    from services.research_direction_service import ResearchDirectionService

    svc = ResearchDirectionService()
    llm = _make_llm(response)

    with (
        patch("services.research_direction_service.get_db_manager") as dbm,
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch("utils.tree_payload.load_latest_tree_payload", return_value=None),
        patch.object(svc, "_read_text", return_value="nội dung về kiến trúc máy tính"),
        patch.object(svc, "_progress"),
    ):
        dbm.return_value.session.return_value = _session(catalog_names)
        result = await svc._do_extract("DOC_001", extraction_id=None)
    return result, llm


class TestPromptIsOpenEnded:
    @pytest.mark.asyncio
    async def test_model_is_not_confined_to_the_catalog(self):
        _, llm = await _run("[]")
        prompt = llm.chat_completion.await_args.args[0]

        assert "MUST match a catalog entry" not in prompt
        assert "Do NOT invent" not in prompt

    @pytest.mark.asyncio
    async def test_model_is_asked_to_propose_from_its_own_knowledge(self):
        _, llm = await _run("[]")
        prompt = llm.chat_completion.await_args.args[0].lower()

        assert "propose" in prompt
        assert "own knowledge" in prompt

    @pytest.mark.asyncio
    async def test_proposals_must_still_relate_to_the_document(self):
        """Ungrounded in a catalog, not ungrounded in the document."""
        _, llm = await _run("[]")
        prompt = llm.chat_completion.await_args.args[0]

        assert "nội dung về kiến trúc máy tính" in prompt

    @pytest.mark.asyncio
    async def test_catalog_is_offered_as_context_not_as_a_menu(self):
        _, llm = await _run("[]")
        prompt = llm.chat_completion.await_args.args[0]

        assert "Trí tuệ nhân tạo và khoa học dữ liệu" in prompt
        assert "beyond this list" in prompt

    @pytest.mark.asyncio
    async def test_absent_catalog_does_not_break_the_prompt(self):
        _, llm = await _run("[]", catalog_names=[])
        prompt = llm.chat_completion.await_args.args[0]

        assert "(empty catalog)" not in prompt


class TestPredefinedIsDecidedByUs:
    @pytest.mark.asyncio
    async def test_name_in_catalog_is_marked_predefined_whatever_the_model_said(self):
        response = (
            '[{"direction_name": "Trí tuệ nhân tạo và khoa học dữ liệu", '
            '"is_predefined": false, "confidence": 0.9}]'
        )
        result, _ = await _run(response)

        assert result["directions"][0]["is_predefined"] is True

    @pytest.mark.asyncio
    async def test_near_miss_spelling_collapses_onto_the_catalog_entry(self):
        """Otherwise every spelling variant becomes its own permanent row."""
        response = (
            '[{"direction_name": "Kỹ thuật ra đa - dẫn đường", '
            '"is_predefined": false, "confidence": 0.8}]'
        )
        result, _ = await _run(response)

        assert result["directions"][0]["direction_name"] == "Kỹ thuật ra đa – dẫn đường"
        assert result["directions"][0]["is_predefined"] is True

    @pytest.mark.asyncio
    async def test_novel_direction_is_stored_as_new_even_if_the_model_claims_otherwise(self):
        response = (
            '[{"direction_name": "Kiến trúc bộ nhớ phi tuần tự cho suy luận LLM", '
            '"is_predefined": true, "confidence": 0.7}]'
        )
        result, _ = await _run(response)

        assert result["directions"][0]["is_predefined"] is False
