"""5 of the 20 keywords in §2.3 of N4.11.160 were unusable.

Two distinct faults:

* **Duplicated pair** — `Core i7 (Core i7)`, `OMAP4430 (OMAP4430)`, `ATmega168
  (ATmega168)`. The display form is "Vietnamese (original)"; for proper nouns and
  part numbers both halves are equal and the repeated parenthesis is noise.

* **Bibliographic keyword** — `Kiến trúc máy tính (АРХИТЕКТУРА КОМПЬЮТЕРА)` is
  the **book title** and `E. Tanenbaum, T. Austin` are the **authors**. §1 has
  both. The keywords stage runs in parallel with biblio so it cannot know; digest
  assembly is the only place that knows both.
"""

import pytest

from utils.digest_format import collapse_bilingual_display, drop_bibliographic_keywords


class TestCollapseBilingualDisplay:
    @pytest.mark.parametrize(
        "display,expected",
        [
            ("Core i7 (Core i7)", "Core i7"),
            ("OMAP4430 (OMAP4430)", "OMAP4430"),
            ("FPGA (FPGA)", "FPGA"),
            ("  ATmega168  ( ATmega168 )  ", "ATmega168"),
            (
                "LUT (lut)",
                "LUT",
            ),
        ],
    )
    def test_identical_halves_collapse(self, display, expected):
        assert collapse_bilingual_display(display) == expected

    @pytest.mark.parametrize(
        "display",
        [
            "Bộ nhớ ảo (виртуальной памяти)",
            "Kiến trúc máy tính (Computer architecture)",
            "Mạch tích hợp (интегральные схемы)",
        ],
    )
    def test_real_bilingual_pairs_are_kept(self, display):
        assert collapse_bilingual_display(display) == display

    def test_no_parenthesis_is_untouched(self):
        assert collapse_bilingual_display("Bộ nhớ ảo") == "Bộ nhớ ảo"

    def test_nested_or_multiple_parens_are_left_alone(self):
        """Only the unambiguous `X (Y)` form is handled — leave the rest alone."""
        text = "Chuẩn ADSL (Asymmetric DSL) (ITU G.992.1)"
        assert collapse_bilingual_display(text) == text

    def test_empty_is_safe(self):
        assert collapse_bilingual_display("") == ""
        assert collapse_bilingual_display(None) == ""


class TestDropBibliographicKeywords:
    BIB = {
        "title_display": "Архитектура компьютера (Structured Computer Organization)",
        "authors": "Э. Таненбаум, Т. Остин",
    }

    def _names(self, kws):
        return [k["display"] for k in kws]

    def test_title_keyword_is_dropped(self):
        kws = [
            {
                "keyword": "Архитектура компьютера",
                "display": "Kiến trúc máy tính (АРХИТЕКТУРА КОМПЬЮТЕРА)",
            },
            {"keyword": "виртуальной памяти", "display": "bộ nhớ ảo (виртуальной памяти)"},
        ]

        assert self._names(drop_bibliographic_keywords(kws, self.BIB)) == [
            "bộ nhớ ảo (виртуальной памяти)"
        ]

    def test_author_keyword_is_dropped(self):
        kws = [
            {
                "keyword": "Э. ТАНЕНБАУМ, Т. ОСТИН",
                "display": "E. Tanenbaum, T. Austin (Э. ТАНЕНБАУМ, Т. ОСТИН)",
            },
            {"keyword": "RISC", "display": "hệ thống RISC và CISC (Системы RISC и CISC)"},
        ]

        assert self._names(drop_bibliographic_keywords(kws, self.BIB)) == [
            "hệ thống RISC và CISC (Системы RISC и CISC)"
        ]

    def test_a_single_author_surname_is_enough_to_drop(self):
        kws = [{"keyword": "Таненбаум", "display": "Tanenbaum (Таненбаум)"}]

        assert drop_bibliographic_keywords(kws, self.BIB) == []

    def test_a_genuine_subject_term_survives(self):
        """Do not over-drop: a book's subject overlapping its title is normal."""
        kws = [{"keyword": "многоуровневая организация", "display": "tổ chức máy tính đa cấp"}]

        assert len(drop_bibliographic_keywords(kws, self.BIB)) == 1

    def test_without_a_bibliographic_record_nothing_is_dropped(self):
        kws = [{"keyword": "Архитектура компьютера", "display": "Kiến trúc máy tính"}]

        assert len(drop_bibliographic_keywords(kws, {})) == 1
        assert len(drop_bibliographic_keywords(kws, None)) == 1

    def test_empty_list_is_safe(self):
        assert drop_bibliographic_keywords([], self.BIB) == []


class TestNonCyrillicSources:
    """The corpus is not only Russian — also English, Chinese, Japanese.

    Both of the filter's assumptions are Latin/Cyrillic-shaped: splitting an
    author list on comma/`и`/`and`, and banning strings of **>= 4 characters**.
    For CJK both are wrong — `田中太郎、山田花子` carries no separator the filter
    knows, and a Chinese proper noun (`王小明`) is only 3 characters.
    """

    def _names(self, kws):
        return [k["display"] for k in kws]

    def test_japanese_authors_split_on_the_ideographic_comma(self):
        bib = {"title_display": "コンピュータの構成と設計", "authors": "田中太郎、山田花子"}
        kws = [
            {"keyword": "山田花子", "display": "Yamada Hanako (山田花子)"},
            {"keyword": "仮想記憶", "display": "bộ nhớ ảo (仮想記憶)"},
        ]

        assert self._names(drop_bibliographic_keywords(kws, bib)) == ["bộ nhớ ảo (仮想記憶)"]

    def test_a_three_character_chinese_name_is_short_but_still_a_name(self):
        bib = {"title_display": "计算机组成与设计", "authors": "王小明，李华"}
        kws = [
            {"keyword": "王小明", "display": "Vương Tiểu Minh (王小明)"},
            {"keyword": "李华", "display": "Lý Hoa (李华)"},
        ]

        assert drop_bibliographic_keywords(kws, bib) == []

    def test_a_katakana_name_split_on_the_middle_dot(self):
        bib = {
            "title_display": "コンピュータアーキテクチャ",
            "authors": "アンドリュー・タネンバウム",
        }
        kws = [{"keyword": "タネンバウム", "display": "Tanenbaum (タネンバウム)"}]

        assert drop_bibliographic_keywords(kws, bib) == []

    def test_a_genuine_cjk_subject_term_survives(self):
        bib = {"title_display": "计算机组成与设计", "authors": "王小明，李华"}
        kws = [
            {"keyword": "流水线", "display": "kỹ thuật đường ống (流水线)"},
            {"keyword": "缓存一致性", "display": "nhất quán bộ nhớ đệm (缓存一致性)"},
        ]

        assert len(drop_bibliographic_keywords(kws, bib)) == 2

    def test_a_single_cjk_character_is_never_banned(self):
        """A 2-character floor for CJK: a single character is too common to ban."""
        bib = {"title_display": "机", "authors": "李"}
        kws = [{"keyword": "机", "display": "máy (机)"}]

        assert len(drop_bibliographic_keywords(kws, bib)) == 1

    def test_full_width_parentheses_collapse_too(self):
        """CJK keyboards emit `（）` rather than `()` — the same duplicated pair."""
        assert collapse_bilingual_display("OMAP4430（OMAP4430）") == "OMAP4430"
        assert collapse_bilingual_display("流水线（流水线）") == "流水线"

    def test_a_full_width_bilingual_pair_is_kept(self):
        text = "kỹ thuật đường ống（流水线）"
        assert collapse_bilingual_display(text) == text

    def test_english_authors_keep_the_four_character_threshold(self):
        """Latin keeps the stricter floor: `Tom`, `and`, and initials must not ban."""
        bib = {
            "title_display": "Structured Computer Organization",
            "authors": "Andrew S. Tanenbaum and Todd Austin",
        }
        kws = [
            {"keyword": "Tanenbaum", "display": "Tanenbaum"},
            {"keyword": "bus", "display": "bus hệ thống (bus)"},
            {"keyword": "RISC", "display": "kiến trúc RISC (RISC)"},
        ]

        assert self._names(drop_bibliographic_keywords(kws, bib)) == [
            "bus hệ thống (bus)",
            "kiến trúc RISC (RISC)",
        ]
