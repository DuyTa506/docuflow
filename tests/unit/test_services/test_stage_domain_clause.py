"""Every digest stage must put the document's domain and termbase in its prompt."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.domain_context import DomainContext

CTX = DomainContext(
    {"specialty": {"vi": "Đạn đạo học", "en": "Ballistics"}},
    {"Shaped Charges": {"vi": "Đạn lõm", "en": "Shaped charge"}},
)


class TestKeywords:
    def test_rerank_instructions_carry_the_domain(self):
        from services.keyword_service import KeywordService

        text = KeywordService._rerank_instructions(10, domain="DOCUMENT DOMAIN: Đạn đạo học")

        assert "DOCUMENT DOMAIN: Đạn đạo học" in text

    @pytest.mark.asyncio
    async def test_refine_puts_the_termbase_hit_in_the_prompt(self):
        from services.keyword_service import KeywordService

        svc = KeywordService()
        llm = MagicMock()
        prompts = []

        async def chat(_llm, prompt):
            prompts.append(prompt)
            return "[]"

        with (
            patch("services.keyword_service.load_domain_context", return_value=CTX),
            patch("utils.prompt_budget.allocate_document_sample", return_value=("excerpt", {})),
            patch.object(svc, "_chat_keyword_rerank", side_effect=chat),
            patch.object(
                svc,
                "_parse_rerank_response",
                new=AsyncMock(return_value=([{"keyword": "Shaped Charges"}], False)),
            ),
            patch.object(svc, "_progress"),
            patch(
                "utils.keyword_validation.validate_keyword_batch",
                return_value=([{"keyword": "Shaped Charges"}], {"min_required": 0}),
            ),
        ):
            await svc._refine_keywords(
                llm,
                candidates=[],
                candidate_lines="1. Shaped Charges",
                document_id="DOC_001",
                text="the Shaped Charges chapter",
                pool_size=10,
                task_id=None,
            )

        assert "Đạn đạo học" in prompts[0]
        assert "Shaped Charges → Đạn lõm" in prompts[0]


@pytest.fixture
def current_domain():
    """Make CTX the running document's context, and restore it afterwards."""
    from services import domain_context

    token = domain_context._current.set(CTX)
    yield CTX
    domain_context._current.reset(token)


def _llm(answer="Tóm tắt chương."):
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=answer)
    llm.count_tokens = MagicMock(side_effect=lambda text: max(1, len(text) // 4))
    llm.encoding = None
    llm.extract_json = MagicMock(return_value=[{"n": 1, "title_vi": "Đạn lõm"}])
    return llm


class TestMainContent:
    @pytest.mark.asyncio
    async def test_chapter_prompt_has_the_domain_and_the_termbase_hit(self, current_domain):
        from services.main_content_service import MainContentService

        llm = _llm()
        text = "The Shaped Charges jet penetrates armour. " * 30
        node = {
            "title": "Shaped Charges",
            "content": "",
            "children": [{"content": text, "children": []}],
        }

        await MainContentService()._summarize_chapter(llm, node, 18)

        prompt = llm.chat_completion.await_args.args[0]
        assert "Đạn đạo học" in prompt and "Shaped Charges → Đạn lõm" in prompt

    @pytest.mark.asyncio
    async def test_title_translation_is_told_the_house_term(self, current_domain):
        from services.main_content_service import MainContentService

        llm = _llm('[{"n": 1, "title_vi": "Đạn lõm"}]')
        chapters = [{"title_original": "Shaped Charges", "title_vi": "Shaped Charges"}]

        await MainContentService()._translate_titles(llm, chapters)

        prompt = llm.chat_completion.await_args.args[0]
        assert "Shaped Charges → Đạn lõm" in prompt

    @pytest.mark.asyncio
    async def test_no_context_leaves_the_prompt_as_it_was(self):
        from services.main_content_service import MainContentService

        llm = _llm()
        text = "The Shaped Charges jet penetrates armour. " * 30
        node = {
            "title": "Shaped Charges",
            "content": "",
            "children": [{"content": text, "children": []}],
        }

        await MainContentService()._summarize_chapter(llm, node, 18)

        assert "DOCUMENT DOMAIN" not in llm.chat_completion.await_args.args[0]


class TestSummarization:
    @pytest.mark.asyncio
    async def test_the_abstract_prompt_has_the_domain_and_the_termbase_hit(self, current_domain):
        from services.summarization_service import compose_document_summary

        llm = _llm("Tài liệu trình bày đạn đạo học.")
        tree = {
            "title": "Doc",
            "children": [
                {"title": "Shaped Charges", "summary": "The Shaped Charges jet.", "children": []},
                {"title": "Wound Ballistics", "summary": "Wounds.", "children": []},
                {"title": "Trajectories", "summary": "Flight.", "children": []},
            ],
        }

        await compose_document_summary(llm, tree)

        prompt = llm.chat_completion.await_args.args[0]
        assert "Đạn đạo học" in prompt and "Shaped Charges → Đạn lõm" in prompt


class _Stop(Exception):
    """Raised by the spy once the prompt prefix has been captured."""


def _prefix_spy(captured):
    def spy(*, fixed_parts, **_):
        captured.append(fixed_parts[0])
        raise _Stop

    return spy


class TestResearchDirectionsAndUsageScope:
    @pytest.mark.asyncio
    async def test_research_directions_prompt_names_the_specialty(self):
        from services.research_direction_service import ResearchDirectionService

        svc, captured = ResearchDirectionService(), []
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        session.query.return_value.filter.return_value.all.return_value = []
        with (
            patch("services.domain_context.load_domain_context", return_value=CTX),
            patch("services.research_direction_service.get_db_manager") as dbm,
            patch("api.dependencies.get_llm_client", return_value=MagicMock()),
            patch.object(svc, "_read_text", return_value="text"),
            patch.object(svc, "_progress"),
            patch("utils.prompt_budget.allocate_document_sample", _prefix_spy(captured)),
        ):
            dbm.return_value.session.return_value = session
            with pytest.raises(_Stop):
                await svc._do_extract("DOC_001", extraction_id=None)

        assert "Đạn đạo học" in captured[0]

    @pytest.mark.asyncio
    async def test_usage_scope_prompt_names_the_specialty(self):
        from services.usage_scope_service import UsageScopeService

        svc, captured = UsageScopeService(), []
        with (
            patch("services.domain_context.load_domain_context", return_value=CTX),
            patch("services.usage_scope_service.get_db_manager"),
            patch("api.dependencies.get_llm_client", return_value=MagicMock()),
            patch.object(svc, "_read_text", return_value="text"),
            patch.object(svc, "_progress"),
            patch("utils.prompt_budget.allocate_document_sample", _prefix_spy(captured)),
        ):
            with pytest.raises(_Stop):
                await svc._extract("DOC_001")

        assert "Đạn đạo học" in captured[0]


class TestBibliographicTitle:
    async def _run(self, answer, source_language="en"):
        import json

        from services.bibliographic_service import BibliographicService

        svc = BibliographicService()
        llm = MagicMock()
        llm.chat_completion = AsyncMock(return_value=json.dumps(answer))
        llm.extract_json = json.loads
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        doc = MagicMock(total_pages=502, title="Ballistics", source_language=source_language)
        saved = MagicMock()
        session.query.return_value.filter.return_value.first.return_value = saved
        with (
            patch("services.domain_context.load_domain_context", return_value=CTX),
            patch("services.bibliographic_service.get_db_manager") as dbm,
            patch("data.repositories.DocumentRepository") as repo,
            patch("api.dependencies.get_llm_client", return_value=llm),
            patch.object(svc, "_read_front_matter", return_value="Shaped Charges front matter"),
            patch.object(svc, "_progress"),
            patch("utils.prompt_budget.allocate_document_sample", return_value=("excerpt", {})),
        ):
            dbm.return_value.session.return_value = session
            repo.return_value.get.return_value = doc
            result = await svc._extract("DOC_001")
        return result, llm.chat_completion.await_args.args[0]

    @pytest.mark.asyncio
    async def test_title_is_assembled_by_code_in_the_agreed_order(self):
        result, _ = await self._run(
            {
                "title_original": "Ballistics: Theory and Design of Guns and Ammunition",
                "title_vi": "Đạn đạo học: Lý thuyết và thiết kế súng, đạn dược",
                "title_en": "Ballistics: Theory and Design of Guns and Ammunition",
            }
        )

        assert result["title_display"] == (
            "Ballistics: Theory and Design of Guns and Ammunition "
            "(Đạn đạo học: Lý thuyết và thiết kế súng, đạn dược)"
        )

    @pytest.mark.asyncio
    async def test_prompt_names_the_specialty_so_the_title_is_not_word_by_word(self):
        _, prompt = await self._run({"title_original": "Ballistics"})

        assert "Đạn đạo học" in prompt

    @pytest.mark.asyncio
    async def test_an_old_style_title_display_answer_still_works(self):
        result, _ = await self._run({"title_display": "Radar (Ra-đa)"})

        assert result["title_display"] == "Radar (Ra-đa)"


class TestKeywordDisplay:
    def _display(self, item, glossary=None, lang="en"):
        from services.keyword_service import KeywordService

        return KeywordService._display_for(item, glossary or {}, lang)

    def test_the_termbase_rendering_wins_over_the_models_own(self):
        out = self._display(
            {"keyword": "Interior ballistics", "vi": "Nội đạn học", "en": "Interior ballistics"},
            {"interior ballistics": "Đạn đạo trong"},
        )

        assert out == "Đạn đạo trong (Interior ballistics)"

    def test_the_model_rendering_is_used_when_the_termbase_has_no_entry(self):
        out = self._display(
            {"keyword": "Wind effects", "vi": "Ảnh hưởng của gió", "en": "Wind effects"}
        )

        assert out == "Ảnh hưởng của gió (Wind effects)"

    def test_without_a_vietnamese_piece_the_models_display_is_kept(self):
        out = self._display({"keyword": "radar", "display": "Ra-đa (radar)"})

        assert out == "Ra-đa (radar)"
