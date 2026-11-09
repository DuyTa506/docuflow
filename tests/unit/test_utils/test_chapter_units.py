"""§2.2 units were "whatever the tree root happened to have as children".

On N4.11.160 (816-page Russian book, 9 chapters + 3 appendices) that produced
265 entries: table-of-contents lines, table and listing captions, a bare
formula, and seven publisher-advertisement blocks — each expanded by the LLM
into a ~150-word academic summary, for a 76-page digest.

The tree's *hierarchy* is noise (levels come from a fixed percentile cut), but
its *reading order* is exact. So units are contiguous reading-order spans
between structural anchors, which is immune to every level defect at once —
including chapters whose subsections ended up as their siblings.
"""

import pytest

from config.settings import settings
from utils.chapter_units import effective_max_units, select_chapter_units


def _node(title, content="", label="text", page=1):
    return {"title": title, "content": content, "page_number": page, "children": []}


def _tree(children):
    return {"title": "Document", "children": children}


BODY = "Текст раздела о микроархитектуре и конвейерной обработке команд. " * 40  # ~2.5k chars


def _russian_book():
    """265 flat nodes: front matter, TOC block, 9 chapters, 3 appendices, ads."""
    children = [_node("Т 18 Архитектура компьютера", "6-е изд. СПб.: Питер, 2013.")]
    # A table-of-contents block: short dot-leader lines.
    children += [_node(f"Глава {i}. Раздел ..... {i * 80}") for i in range(1, 11)]

    for chapter in range(1, 10):
        children.append(_node(f"Глава {chapter}. Организация систем", "", page=chapter * 80))
        for sub in range(1, 21):
            children.append(
                _node(f"{chapter}.{sub} Подраздел о процессорах", BODY, page=chapter * 80 + sub)
            )
        # Junk the thinning bug injects: body text promoted to a node title.
        children.append(_node("F = (( JAMZ И Z ) ИЛИ ( JAMN И N )) ИЛИ NEXT_ADDRESS [8]", BODY))
        children.append(_node("Таблица 3.5. Дополнительные сигналы шины PCI", BODY))

    for letter in ("А", "Б", "В"):
        children.append(_node(f"Приложение {letter}", "", page=700))
        for sub in range(1, 6):
            children.append(_node(f"Раздел приложения {sub}", BODY, page=700 + sub))

    # Publisher back matter.
    children += [
        _node("ЗАКАЗАТЬ КНИГИ ИЗДАТЕЛЬСКОГО ДОМА «ПИТЕР»", "Наложенным платежом.", page=815),
        _node("РОССИЯ", "Санкт-Петербург.", page=815),
        _node("УКРАИНА", "Киев, Харьков.", page=816),
    ]
    return _tree(children)


class TestChapterVocabularyTier:
    def test_russian_book_yields_one_unit_per_real_chapter(self):
        units, meta = select_chapter_units(_russian_book())

        assert meta["unit_selection_tier"] == "chapter_vocabulary"
        assert len(units) == 12, [u["title"] for u in units]
        assert units[0]["title"].startswith("Глава 1")
        assert units[8]["title"].startswith("Глава 9")
        assert units[11]["title"].startswith("Приложение В")
        assert meta["coverage_ratio"] > 0.9

    def test_junk_nodes_never_become_units(self):
        units, _ = select_chapter_units(_russian_book())
        titles = " | ".join(u["title"] for u in units)

        assert "JAMZ" not in titles, "a formula became a chapter"
        assert "Таблица" not in titles, "a table caption became a chapter"
        assert "ПИТЕР" not in titles, "publisher back matter became a chapter"
        assert "....." not in titles, "a table-of-contents line became a chapter"

    def test_absorbed_back_matter_keeps_its_text(self):
        """Filtering removes entries, never content."""
        units, _ = select_chapter_units(_russian_book())
        last = units[-1]
        flat = " ".join(c["content"] for c in last["children"])

        assert "Киев" in flat

    def test_units_are_reading_order_spans_not_subtrees(self):
        """A chapter whose subsections are its *siblings* still owns their text."""
        tree = _tree(
            [
                _node("Глава 1. Введение"),
                _node("Уровни", "содержание уровней " * 100),
                _node("Глава 2. Организация"),
                _node("Процессоры", "содержание процессоров " * 100),
            ]
        )

        units, meta = select_chapter_units(tree)

        assert meta["unit_selection_tier"] == "chapter_vocabulary"
        assert len(units) == 2
        assert "содержание уровней" in units[0]["children"][-1]["content"]
        assert "содержание процессоров" in units[1]["children"][-1]["content"]

    def test_toc_duplicate_heading_loses_to_real_heading(self):
        tree = _tree(
            [
                _node("Глава 3. Цифровой логический уровень ..... 210"),
                _node("Глава 3. Цифровой логический уровень", BODY * 3, page=210),
            ]
        )

        units, _ = select_chapter_units(tree)

        assert len(units) == 1
        assert units[0]["page_number"] == 210

    def test_toc_block_run_is_excluded_from_anchors(self):
        toc = [_node(f"Раздел {i} ..... {i * 10}") for i in range(1, 13)]
        tree = _tree(toc + [_node("Глава 1. Введение", BODY * 4)])

        units, meta = select_chapter_units(tree)

        assert len(units) == 1
        assert units[0]["title"].startswith("Глава 1")

    def test_thinning_artifact_titles_never_anchor(self):
        long_body_as_title = "ПРИ ОФОРМЛЕНИИ ЗАКАЗА УКАЖИТЕ адрес доставки и телефон. " * 4
        tree = _tree(
            [
                _node("Глава 1. Введение", BODY * 3),
                _node(long_body_as_title, long_body_as_title),
                _node("Глава 2. Организация", BODY * 3),
            ]
        )

        units, _ = select_chapter_units(tree)

        assert [u["title"] for u in units] == ["Глава 1. Введение", "Глава 2. Организация"]


class TestFallbackTiers:
    def test_numbered_sections_tier(self):
        tree = _tree(
            [_node(f"{i}. Методы исследования", BODY * 2) for i in range(1, 6)],
        )

        units, meta = select_chapter_units(tree)

        assert meta["unit_selection_tier"] == "numbered_sections"
        assert len(units) == 5

    def test_root_children_tier_preserves_clean_docx_tree(self):
        tree = _tree(
            [
                {
                    "title": f"Phần {i}",
                    "content": BODY * 2,
                    "page_number": i,
                    "children": [_node(f"Noi dung {i}.1", BODY)],
                }
                for i in range(1, 7)
            ]
        )
        # Strip the vocabulary signal so the text-evidence tiers cannot fire.
        for child in tree["children"]:
            child["title"] = child["title"].replace("Phần ", "Chuong so ")

        units, meta = select_chapter_units(tree)

        assert meta["unit_selection_tier"] == "root_children"
        assert len(units) == 6

    @staticmethod
    def _paper(extra_roots=()):
        """A journal article: the title is the tree's only root, its sections hang under it.

        The Nội dung 4 set had 11 of 52 articles in this shape (up to 216k chars);
        each came out as ONE §2.2 entry — the whole paper in 5–8 sentences.
        """
        # Deliberately not the canonical IMRaD names — those are read by the
        # article_sections tier, and this shape must work without them.
        sections = ["TUTOR DESIGN", "LEARNER PROFILES", "STUDY SETUP", "OUTCOMES", "LIMITATIONS"]
        paper = {
            "title": "Intelligent Tutors for Adult Learners",
            "content": "Abstract. " + BODY[:600],
            "page_number": 1,
            "children": [
                _node(name, f"{name} text. " + BODY * 2, page=i + 1)
                for i, name in enumerate(sections)
            ],
        }
        return _tree([paper, *extra_roots])

    def test_paper_whose_only_root_is_its_title_splits_into_its_sections(self):
        units, meta = select_chapter_units(self._paper())

        assert meta["unit_selection_tier"] == "root_children"
        titles = [u["title"] for u in units]
        assert len(units) >= 5, titles
        for name in ("TUTOR DESIGN", "STUDY SETUP", "LIMITATIONS"):
            assert any(name in t for t in titles), titles

    def test_paper_with_a_trailing_references_root_still_splits(self):
        units, _ = select_chapter_units(
            self._paper([_node("REFERENCES", "[1] A. Author. 2020. " * 80)])
        )

        assert len(units) >= 5, [u["title"] for u in units]

    def test_descending_into_sections_keeps_every_character(self):
        tree = self._paper()
        units, _ = select_chapter_units(tree)
        flat = " ".join(c["content"] for u in units for c in u["children"])

        assert "Abstract." in flat
        for name in ("TUTOR DESIGN", "LEARNER PROFILES", "STUDY SETUP", "OUTCOMES", "LIMITATIONS"):
            assert f"{name} text." in flat

    def test_bare_book_title_does_not_count_against_the_cap(self):
        """百戰奇略: a title-only root over exactly `max_units` 卷.

        The layout tree repeats a heading's title as its content, so "title-only"
        means content equal to the title, not empty content.

        Counting the title as a start made 11 > 10, the tier declined, and mass
        segmentation cut the ten 卷 into four machine-cut §2.2 entries (ND4 run 5).
        """
        pian = "凡用兵之道，以計為首。未戰之時，先料將之賢愚，敵之強弱。" * 10
        juan = [
            {
                "title": f"卷{j:02d}",
                "content": f"卷{j:02d}",
                "page_number": j,
                "children": [_node(f"篇第{j}{k}", pian, page=j) for k in range(10)],
            }
            for j in range(1, 11)
        ]
        tree = _tree([{"title": "百戰奇略", "content": "百戰奇略", "page_number": 1, "children": juan}])

        units, meta = select_chapter_units(tree)

        assert meta["max_units"] == 10
        assert meta["unit_selection_tier"] == "root_children"
        assert [u["title"] for u in units] == [f"卷{j:02d}" for j in range(1, 11)]

    def test_long_paper_with_no_usable_headings_is_cut_by_mass_not_left_whole(self):
        """Layout found no headings: the title node plus body sentences as node titles.

        The DOC_045 shape — 81k chars under the title, its "sections" titled
        "For military logistics, Copilot suggested the following: …".
        """
        sentence = "For military logistics, Copilot suggested the following applications " * 3
        paper = {
            "title": "On the Military Applications of Large Language Models",
            "content": BODY * 8,
            "page_number": 1,
            "children": [_node(f"{sentence}{i}", BODY * 8, page=i + 2) for i in range(3)],
        }

        units, meta = select_chapter_units(_tree([paper]))

        assert meta["unit_selection_tier"] == "mass_segmentation"
        assert len(units) >= 3, len(units)

    def test_mass_segmentation_when_no_headings_exist(self):
        """Every node is body text used as its own title (the DOC_020 shape)."""
        tree = _tree([_node(BODY[:400], BODY[:400]) for _ in range(40)])

        units, meta = select_chapter_units(tree)

        assert meta["unit_selection_tier"] == "mass_segmentation"
        assert 1 <= len(units) <= 3, len(units)

    def test_empty_and_single_node_trees(self):
        assert select_chapter_units({})[0] == []
        assert select_chapter_units({"title": "Doc", "children": []})[0] == []
        units, _ = select_chapter_units(_tree([_node("Only", BODY)]))
        assert len(units) == 1


class TestAdaptiveCap:
    """The §2.2 ceiling must scale with the document, not be one fixed number.

    A 40-page report and an 816-page book cannot share a cap: fixed at 25 it is
    far too loose for the report and needlessly tight for a multi-volume work.
    Growth is sub-linear — a document twice as long does not warrant twice as
    many digest entries.
    """

    def test_cap_grows_with_document_length(self):
        small = effective_max_units(120_000)  # ~50 pages
        book = effective_max_units(1_000_000)  # ~400 pages
        huge = effective_max_units(2_400_000)  # ~1000 pages

        assert small < book < huge

    def test_cap_growth_is_sublinear(self):
        assert effective_max_units(4_000_000) < 4 * effective_max_units(1_000_000)

    def test_reference_book_anchors_the_curve(self):
        assert effective_max_units(1_000_000) == 25

    def test_cap_has_floor_for_tiny_documents(self):
        """On short documents MIN_UNIT_CHARS does the anti-fragmentation work,
        so the ceiling must stay loose enough not to squeeze real sections."""
        assert effective_max_units(500) == effective_max_units(0) == 10

    def test_cap_has_ceiling_for_enormous_documents(self):
        assert effective_max_units(50_000_000) == 40

    def test_explicit_setting_overrides_the_curve(self, monkeypatch):
        monkeypatch.setattr(settings, "main_content_max_units", 12)

        assert effective_max_units(1_000_000) == 12

    def test_meta_reports_the_cap_actually_applied(self):
        tree = _tree([_node(f"Глава {i}. Тема", BODY * 2) for i in range(1, 6)])

        _, meta = select_chapter_units(tree)

        assert meta["max_units"] == effective_max_units(sum(len(BODY * 2) for _ in range(5)))


class TestBounds:
    def test_tiny_units_merge_into_previous_never_dropped(self):
        tree = _tree(
            [
                _node("Глава 1. Введение", BODY * 3),
                _node("Приложение А", "крошечный текст"),
            ]
        )

        units, meta = select_chapter_units(tree)

        assert len(units) == 1, "a sub-page appendix must be absorbed, not listed"
        flat = " ".join(c["content"] for c in units[0]["children"])
        assert "крошечный текст" in flat
        assert meta["units_before_merge"] == 2

    def test_never_exceeds_max_units(self):
        tree = _tree(
            [
                node
                for i in range(1, 101)
                for node in (_node(f"Глава {i}. Тема"), _node(f"Раздел {i}", BODY * 2))
            ]
        )

        units, meta = select_chapter_units(tree)

        assert len(units) <= meta["max_units"]
        assert len(units) == meta["max_units"]

    def test_coverage_ratio_rejects_late_first_anchor(self):
        """A heading matched deep in the body must not define the whole document."""
        tree = _tree(
            [_node(f"Параграф {i}", BODY * 2) for i in range(1, 10)]
            + [_node("Глава 1. Заключение", BODY)]
        )

        _, meta = select_chapter_units(tree)

        assert meta["unit_selection_tier"] != "chapter_vocabulary"

    @pytest.mark.parametrize("ratio", [0.0, 0.5])
    def test_prologue_kept_only_when_substantial(self, ratio):
        prologue_body = BODY * 8 if ratio else "короткое предисловие"
        tree = _tree(
            [
                _node("Предисловие", prologue_body),
                _node("Глава 1. Введение", BODY * 4),
                _node("Глава 2. Организация", BODY * 4),
            ]
        )

        units, _ = select_chapter_units(tree)

        titles = [u["title"] for u in units]
        if ratio:
            assert titles[0] == "Предисловие"
        else:
            assert titles[0].startswith("Глава 1")


class TestUnitAnchoredOnABareLabel:
    """`Приложение Б.` is a heading with no name, and §2.2 printed it as such.

    Observed on N4.11.160: the appendix rendered as a bare `Phụ lục B.` — a
    label, a full stop, and nothing else — while the book's own table of
    contents reads `Приложение Б. Числа с плавающей точкой`. The name was never
    missing from the document; it sits on the next heading line, because the
    extraction split the two-line heading into two nodes.

    The name is taken from the unit's own next member, so it is quoted from the
    document rather than invented. A member that is itself a structural label
    is not a name, and a unit that already has a name is left alone.
    """

    @staticmethod
    def _appendix(anchor, first_member="Числа с плавающей точкой"):
        return _tree(
            [
                _node("Глава 1. Введение", "", page=1),
                _node("1.1 Раздел", BODY, page=2),
                _node(anchor, "", page=700),
                _node(first_member, BODY, page=701),
                _node("Стандарт IEEE 754", BODY, page=702),
            ]
        )

    def test_the_name_comes_from_the_next_member(self):
        units, _ = select_chapter_units(self._appendix("Приложение Б."))

        assert units[-1]["title"] == "Приложение Б. Числа с плавающей точкой"

    def test_a_unit_that_already_has_a_name_is_untouched(self):
        units, _ = select_chapter_units(self._appendix("Приложение Б. Числа с плавающей точкой"))

        assert units[-1]["title"] == "Приложение Б. Числа с плавающей точкой"

    def test_junk_promoted_to_a_title_is_not_borrowed_as_a_name(self):
        """Thinning promotes body text to node titles — a formula is not a name.

        `anchor_eligible` already encodes "this title is a real heading", which
        is the same question, so the borrowed name reuses it rather than
        inventing a second rule.
        """
        units, _ = select_chapter_units(
            self._appendix("Приложение Б.", "F = (( JAMZ И Z ) ИЛИ ( JAMN И N )) ИЛИ NEXT" * 5)
        )

        assert units[-1]["title"] == "Приложение Б. Стандарт IEEE 754"

    def test_the_borrowed_name_is_not_lost_from_the_content(self):
        """Adopting the title must not remove that member's text from the unit."""
        units, _ = select_chapter_units(self._appendix("Приложение Б."))

        titles = [c["title"] for c in units[-1]["children"]]
        assert "Числа с плавающей точкой" in titles


VI_BODY = "Kết quả khảo sát cho thấy năng lực công nghệ thông tin của giảng viên. " * 40
RU_BODY = "Результаты исследования показывают роль цифровых компетенций. " * 40


class TestArticleSectionsTier:
    """A journal paper's outline is usually unnumbered — «MỞ ĐẦU … KẾT LUẬN»,
    «Введение … Заключение» — and no other tier reads it. In ND4 run 4
    (DOC_229) the author affiliations «1 Trường…», «2 Trường…» won
    numbered_sections instead, and a five-section paper got two §2.2 entries."""

    def _vietnamese_paper(self):
        return _tree(
            [
                _node("TÓMTẮT", "Bài báo đánh giá năng lực của nữ giảng viên."),
                _node("1 Trường Đại học Luật Thành phố Hồ Chí Minh, Việt Nam", "Liên hệ."),
                _node("2 Trường Đại học Khoa học Xã hội và Nhân văn", VI_BODY),
                # OCR glues the words of spaced capitals: «MỞĐẦU», «THẢOLUẬN».
                _node("MỞĐẦU", VI_BODY),
                _node("PHƯƠNG PHÁP NGHIÊN CỨU", VI_BODY),
                _node("KẾT QUẢ NGHIÊN CỨU", VI_BODY * 2),
                _node("THẢOLUẬN", VI_BODY),
                _node("KẾT LUẬN", VI_BODY),
                _node("TÀI LIỆU THAM KHẢO", "1. Nguyễn A. Tài liệu. 2020."),
            ]
        )

    def test_vietnamese_paper_gets_one_unit_per_section(self):
        units, meta = select_chapter_units(self._vietnamese_paper())

        assert meta["unit_selection_tier"] == "article_sections"
        titles = [u["title"] for u in units]
        assert titles[-4:] == [
            "PHƯƠNG PHÁP NGHIÊN CỨU",
            "KẾT QUẢ NGHIÊN CỨU",
            "THẢOLUẬN",
            "KẾT LUẬN",
        ], titles
        assert len(units) >= 5

    def test_russian_paper_gets_one_unit_per_section(self):
        tree = _tree(
            [
                _node("Аннотация", "Кратко о работе."),
                _node("Введение", RU_BODY),
                _node("Обзор литературы", RU_BODY),
                _node("Методы", RU_BODY),
                _node("Обсуждение результатов", RU_BODY),
                _node("Заключение", RU_BODY),
            ]
        )
        units, meta = select_chapter_units(tree)

        assert meta["unit_selection_tier"] == "article_sections"
        assert [u["title"] for u in units] == [
            "Введение",
            "Обзор литературы",
            "Методы",
            "Обсуждение результатов",
            "Заключение",
        ]

    def test_two_canonical_headings_are_not_an_outline(self):
        tree = _tree(
            [
                _node("Введение", RU_BODY),
                _node("Боевой путь дивизии", RU_BODY * 3),
                _node("Заключение", RU_BODY),
            ]
        )
        _, meta = select_chapter_units(tree)
        assert meta["unit_selection_tier"] != "article_sections"

    def test_a_numbered_paper_keeps_its_numbered_sections(self):
        body = "The tutoring study measured learner outcomes across sessions. " * 40
        tree = _tree(
            [
                _node("1 Introduction", body),
                _node("2 Related Work", body),
                _node("3 Method", body),
                _node("4 Results", body),
                _node("5 Discussion", body),
                _node("6 Conclusion", body),
            ]
        )
        units, meta = select_chapter_units(tree)
        assert meta["unit_selection_tier"] == "numbered_sections"
        assert len(units) == 6

    def test_a_canonical_heading_that_owns_no_text_does_not_count(self):
        """DOC_235: two-column reading order put «KẾT QUẢ NGHIÊN CỨU» (141 chars)
        before «PHƯƠNG PHÁP», which then swallowed 40k chars of sub-sections."""
        tree = _tree(
            [
                _node("ĐẶT VẤN ĐỀ", VI_BODY),
                _node("KẾT QUẢ NGHIÊN CỨU", "Bảng 1."),
                _node("PHƯƠNG PHÁP NGHIÊN CỨU", VI_BODY),
                _node("Điều luật Nuremberg 1947", VI_BODY * 3),
                _node("Tuyên bố Helsinki 1964", VI_BODY * 3),
            ]
        )
        _, meta = select_chapter_units(tree)
        assert meta["unit_selection_tier"] != "article_sections"


class TestNumberedSectionsNeedAnOutline:
    def test_two_numbered_affiliation_lines_are_not_sections(self):
        """DOC_229 / DOC_203 / DOC_210: «1 Trường Đại học…», «2 Trường…» (or a
        reference «13 ARISA…») won the tier and the paper got two entries."""
        tree = _tree(
            [
                _node("1 Trường Đại học Luật Thành phố Hồ Chí Minh", "Liên hệ."),
                _node("2 Trường Đại học Khoa học Xã hội và Nhân văn", VI_BODY * 4),
                _node("Năng lực số của giảng viên", VI_BODY * 4),
                _node("Thực trạng tại các trường", VI_BODY * 4),
            ]
        )
        _, meta = select_chapter_units(tree)
        assert meta["unit_selection_tier"] != "numbered_sections"


class TestMassSegmentationOfAPaper:
    def test_a_paper_is_cut_at_section_size_not_chapter_size(self):
        """DOC_238 (~47k chars): its outline is thematic («Đối với võ quan»),
        so only mass segmentation applies — and at a 20k-char target a paper
        came out as two §2.2 entries against the references' six."""
        tree = _tree(
            [
                _node("Preface", "Tạp chí Khoa học Xã hội và Nhân văn."),
                _node("TÓMTẮT", "Tóm tắt."),
                {**_node("MỞĐẦU", ""), "children": [_node("Bản quyền", VI_BODY * 3)]},
                {
                    **_node("Chế độ trợ cấp với võ quan và binh lính", ""),
                    "children": [
                        _node("Đối với võ quan", VI_BODY * 3),
                        _node("Đối với binh lính", VI_BODY * 3),
                        _node("Đối với thân nhân của võ quan", VI_BODY * 3),
                    ],
                },
                _node("KẾT LUẬN", VI_BODY * 2),
                _node("ABSTRACT", "Abstract."),
                _node("Correspondence", "Author."),
                _node("History", "Received."),
                _node("Copyright", "CC BY."),
            ]
        )
        units, meta = select_chapter_units(tree)

        assert meta["unit_selection_tier"] == "mass_segmentation"
        assert len(units) >= 4, [u["title"] for u in units]
