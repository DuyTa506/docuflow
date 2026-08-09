"""Which stopword list YAKE gets, and what it costs to get it wrong.

`_yake_language` asked whether the text contains *any* Cyrillic. One quoted
Russian term therefore made a Vietnamese document Russian. Measured on the
Vietnamese prose this pipeline produces — summaries that cite their source terms
in parentheses — Cyrillic is 0.4% of letters and Vietnamese diacritics are 17.0%,
so the two are not close and dominance decides it cleanly. On the Russian source
itself Cyrillic is 96.5%.

The second half is the list that was missing. YAKE ships 35 languages and no
`vi`, so Vietnamese fell back to English stopwords and its own function words
went unfiltered: on that same prose `các` ranked **first of fifty** candidates,
with 11 of 50 at least half function words. YAKE takes a `stopwords` set
directly, which is how `vi` is supplied without a language pack.
"""

import pytest

from services.keyword_service import KeywordService


@pytest.fixture
def service():
    return KeywordService.__new__(KeywordService)


VIETNAMESE_CITING_RUSSIAN = (
    "Kiến trúc máy tính (Архитектура компьютера) trình bày tổ chức của bộ vi xử lý "
    "và các thanh ghi trong hệ thống. Bộ nhớ đệm và đường ống lệnh quyết định hiệu "
    "năng của máy tính khi thực thi các lệnh được nạp từ bộ nhớ chính. "
) * 20

RUSSIAN_CITING_LATIN = (
    "Цифровой логический уровень определяет вентили Core i7 и шину PCI. "
    "Регистр хранит данные, триггер образует память FPGA. "
) * 20

ENGLISH = "Digital logic level defines gates and boolean algebra in registers. " * 20
GREEK = "Η ψηφιακή λογική στάθμη ορίζει πύλες και άλγεβρα Boole στους καταχωρητές. " * 20


class TestDominantScriptWins:
    @pytest.mark.parametrize(
        "text,expected",
        [
            (VIETNAMESE_CITING_RUSSIAN, "vi"),
            (RUSSIAN_CITING_LATIN, "ru"),
            (ENGLISH, "en"),
            (GREEK, "el"),
        ],
        ids=["vietnamese-citing-russian", "russian-citing-latin", "english", "greek"],
    )
    def test_the_script_of_the_bulk_of_the_text_decides(self, service, text, expected):
        assert service._yake_language(text) == expected

    def test_a_single_stray_character_decides_nothing(self, service):
        assert service._yake_language(ENGLISH + " Привет") == "en"

    def test_empty_text_is_safe(self, service):
        assert service._yake_language("") == "en"


class TestVietnameseFunctionWordsAreFiltered:
    JUNK = ("các", "của", "được", "trong", "và")

    def test_a_function_word_is_not_a_candidate(self, service):
        out = [
            c["keyword"].casefold()
            for c in service._content_candidates(VIETNAMESE_CITING_RUSSIAN, max_candidates=30)
        ]

        assert out, "Vietnamese text must not come back empty"
        assert not (set(out) & set(self.JUNK)), [k for k in out if k in self.JUNK]

    def test_real_terms_still_surface(self, service):
        out = " | ".join(
            c["keyword"].casefold()
            for c in service._content_candidates(VIETNAMESE_CITING_RUSSIAN, max_candidates=30)
        )

        assert "máy tính" in out or "kiến trúc" in out, out
