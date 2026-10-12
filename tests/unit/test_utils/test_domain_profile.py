from utils.domain_profile import domain_clause, parse_profile, parse_termbase, termbase_glossary
from utils.glossary import glossary_clause

RAW = {
    "field": {"vi": "Khoa học và kỹ thuật quân sự", "en": "Military science and engineering"},
    "discipline": {"vi": "Vũ khí", "en": "Weapons"},
    "specialty": {"vi": "Đạn đạo học", "en": "Ballistics"},
    "summary_en": "A textbook on interior, exterior and terminal ballistics.",
    "junk": "ignored",
}


class TestParseProfile:
    def test_keeps_the_three_levels_and_drops_unknown_keys(self):
        profile = parse_profile(RAW)

        assert profile["specialty"] == {"vi": "Đạn đạo học", "en": "Ballistics"}
        assert "junk" not in profile

    def test_garbage_gives_an_empty_profile(self):
        assert parse_profile("nope") == {}
        assert parse_profile({"field": 3}) == {}

    def test_a_bare_string_level_is_read_as_both_languages(self):
        profile = parse_profile({"specialty": "Ballistics"})

        assert profile["specialty"] == {"vi": "Ballistics", "en": "Ballistics"}

    def test_clusters_for_a_proceedings_are_kept_and_empty_ones_dropped(self):
        profile = parse_profile(
            {**RAW, "clusters": [{"vi": "Khoa học máy tính", "en": "Computer Science"}, {}, 7]}
        )

        assert profile["clusters"] == [{"vi": "Khoa học máy tính", "en": "Computer Science"}]


class TestDomainClause:
    def test_names_the_field_discipline_and_specialty(self):
        clause = domain_clause(parse_profile(RAW))

        assert (
            "Đạn đạo học" in clause
            and "Vũ khí" in clause
            and "Khoa học và kỹ thuật quân sự" in clause
        )

    def test_no_profile_gives_no_clause(self):
        assert domain_clause({}) == ""
        assert domain_clause(None) == ""


class TestTermbase:
    def test_parses_source_to_vi_and_en(self):
        tb = parse_termbase(
            [
                {"src": "Interior ballistics", "vi": "Đạn đạo trong", "en": "Interior ballistics"},
                {"src": "", "vi": "x"},
                {"src": "same", "vi": "same"},
            ]
        )

        assert tb == {"Interior ballistics": {"vi": "Đạn đạo trong", "en": "Interior ballistics"}}

    def test_missing_en_falls_back_to_the_source_term(self):
        tb = parse_termbase([{"src": "Wound ballistics", "vi": "Đạn đạo vết thương"}])

        assert tb["Wound ballistics"]["en"] == "Wound ballistics"

    def test_glossary_view_feeds_the_existing_prompt_clause(self):
        tb = parse_termbase([{"src": "Shaped charge", "vi": "Đạn lõm"}])

        clause = glossary_clause(termbase_glossary(tb), "the shaped charge jet")

        assert "Shaped charge → Đạn lõm" in clause

    def test_glossary_clause_honours_a_larger_limit(self):
        glossary = {f"term{i}": f"thuật ngữ {i}" for i in range(30)}
        text = " ".join(glossary)

        assert glossary_clause(glossary, text).count("→") == 20
        assert glossary_clause(glossary, text, max_terms=40).count("→") == 30
