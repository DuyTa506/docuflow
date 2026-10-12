"""DOMAIN_PROFILE: one stage that names the document's specialty and pins its terms."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.domain_profile_service import DomainProfileService, outline_titles

PROFILE = {
    "field": {"vi": "Khoa học và kỹ thuật quân sự", "en": "Military science and engineering"},
    "discipline": {"vi": "Vũ khí", "en": "Weapons"},
    "specialty": {"vi": "Đạn đạo học", "en": "Ballistics"},
    "summary_en": "A textbook on ballistics.",
}
TERMS = [
    {"src": "Interior Ballistics", "vi": "Đạn đạo trong", "en": "Interior Ballistics"},
    {"src": "Shaped Charges", "vi": "Đạn lõm", "en": "Shaped charge"},
]
INPUTS = {
    "title": "Ballistics",
    "source_language": "en",
    "outline": ["Interior Ballistics", "Shaped Charges"],
    "excerpt": "Ballistics: theory and design of guns and ammunition",
    "explicit_kind": "book",
}


def _llm(*answers):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(side_effect=list(answers))
    llm.extract_json = lambda text: json.loads(text)
    return llm


async def _run(llm, inputs=INPUTS):
    svc = DomainProfileService()
    saved = {}
    with (
        patch("api.dependencies.get_llm_client", return_value=llm),
        patch.object(svc, "_load_inputs", return_value=inputs),
        patch.object(
            svc, "_save", side_effect=lambda doc, p, t: saved.update(profile=p, termbase=t)
        ),
        patch.object(svc, "_progress"),
    ):
        result = await svc.run_for_pipeline("DOC_001")
    return result, saved


class TestRun:
    @pytest.mark.asyncio
    async def test_saves_the_profile_and_the_termbase(self):
        llm = _llm(json.dumps(PROFILE), json.dumps(TERMS))

        _, saved = await _run(llm)

        assert saved["profile"]["specialty"]["vi"] == "Đạn đạo học"
        assert saved["termbase"]["Shaped Charges"] == {"vi": "Đạn lõm", "en": "Shaped charge"}

    @pytest.mark.asyncio
    async def test_termbase_prompt_carries_the_domain_and_the_headings(self):
        llm = _llm(json.dumps(PROFILE), json.dumps(TERMS))

        await _run(llm)

        terms_prompt = llm.chat_completion.call_args_list[1].args[0]
        assert "Đạn đạo học" in terms_prompt
        assert "Interior Ballistics" in terms_prompt

    @pytest.mark.asyncio
    async def test_an_unparseable_profile_keeps_only_the_kind_and_does_not_raise(self):
        llm = MagicMock()
        llm.chat_completion = AsyncMock(return_value="not json")
        llm.extract_json = MagicMock(side_effect=ValueError("bad"))

        result, saved = await _run(llm)

        assert saved == {"profile": {"doc_kind": "book"}, "termbase": {}}
        assert result["terms"] == 0

    @pytest.mark.asyncio
    async def test_records_the_document_kind_with_the_profile(self):
        llm = _llm(json.dumps(PROFILE), json.dumps(TERMS))

        _, saved = await _run(llm, {**INPUTS, "explicit_kind": "proceedings"})

        assert saved["profile"]["doc_kind"] == "proceedings"


class TestOutlineTitles:
    def test_keeps_top_levels_strips_numbering_and_drops_repeats(self):
        tree = {
            "title": "Doc",
            "children": [
                {
                    "title": "Chapter 1 Interior Ballistics",
                    "children": [
                        {
                            "title": "1.1 Ballistic Disciplines",
                            "children": [{"title": "deep heading that is cut", "children": []}],
                        }
                    ],
                },
                {"title": "Interior Ballistics", "children": []},
            ],
        }

        assert outline_titles(tree, depth=2) == ["Interior Ballistics", "Ballistic Disciplines"]
