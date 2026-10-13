"""§2.2 unit selection on the tree shapes seen in the 2026-09 E2E run.

Every scenario mirrors a real book from E2E_Test (see FINDINGS.md, 3.2 Q1):
the old tiers returned 2 units for an 880-page book, section sentences and
formulas as "chapters", and CompTIA's answer key as its chapter list.
"""

from utils.chapter_units import select_chapter_units

BODY = "The converter operates in continuous conduction mode for this design. " * 40  # ~2.8k


def _node(title, content="", page=1):
    return {"title": title, "content": content, "page_number": page, "children": []}


def _tree(children):
    return {"title": "Document", "children": children}


def _titles(units):
    return [u["title"] for u in units]


class TestSectionPrefixTier:
    """Switching Power Supply Design (880 pp.) has no "Chapter" word at all: a
    chapter is an unnumbered title followed by sections k.1, k.2, …"""

    NAMES = ["Basic Topologies", "Push-Pull Topologies", "Flyback Topologies", "Magnetics"]

    def _book(self):
        children = [_node("Switching Power Supply Design", "Third edition. " * 20)]
        for k, name in enumerate(self.NAMES, start=1):
            children.append(_node("This page intentionally left blank", "", page=k * 40))
            children.append(_node(name, "Intro. " * 5, page=k * 40 + 1))
            for s in range(2, 6):
                children.append(_node(f"{k}.{s} Section about {name}", BODY, page=k * 40 + s))
            # Body text the tree promoted to a title, and the chapter's references.
            children.append(_node("where V = the voltage across the choke", BODY))
            children.append(_node("References", "1. Pressman, A. " * 30))
        return _tree(children)

    def test_one_unit_per_chapter_named_by_its_title(self):
        units, meta = select_chapter_units(self._book())

        assert meta["unit_selection_tier"] == "section_prefix"
        assert _titles(units) == self.NAMES

    def test_each_unit_carries_its_chapter_number(self):
        """Ballistics: §2.2 counted 1…7 by position, then the contents list named
        «Chapter 11» — the section numbers already said which chapter each was."""
        units, _ = select_chapter_units(self._book())

        assert [u.get("chapter_ordinal") for u in units] == [1, 2, 3, 4]

    def test_beats_a_vocabulary_tier_that_finds_fewer_chapters(self):
        tree = self._book()
        tree["children"].insert(1, _node("Part I Topologies", "", page=2))
        units, meta = select_chapter_units(tree)

        assert meta["unit_selection_tier"] == "section_prefix"
        assert len(units) == len(self.NAMES)

    def test_front_matter_name_is_not_a_chapter_title(self):
        """Ru_Book: «Предисловие» sat right before «1.1. Введение»."""
        children = [_node("Предисловие", "Слово к читателю. " * 150)]
        for k in range(1, 4):
            for s in range(1, 5):
                children.append(_node(f"{k}.{s}. Раздел главы {k}", BODY))
        units, _ = select_chapter_units(_tree(children))

        assert "Предисловие" not in _titles(units)[1:]
        assert any(t.startswith("1.1") for t in _titles(units))

    def test_per_paper_numbering_is_not_a_chapter_sequence(self):
        """An edited volume restarts 1., 2., 3. in every paper — not chapters."""
        children = []
        for paper in range(6):
            children.append(_node(f"Paper {paper} on digital twins", "Abstract. " * 30))
            for s in (1, 2, 3):
                children.append(_node(f"{s}.1 Methods of paper {paper}", BODY))
        _, meta = select_chapter_units(_tree(children))

        assert meta["unit_selection_tier"] != "section_prefix"


class TestChapterVocabularyHardening:
    def test_numbering_restarting_per_part_keeps_every_chapter(self):
        """Penetration Testing: A Survival Guide — Module 1..3 each has Chapter 1..10."""
        children = []
        for part in ("I", "II"):
            children.append(_node(f"Part {part}. Module", "", page=1))
            for c in range(1, 4):
                children.append(_node(f"Chapter {c}. Topic {part}-{c}", BODY * 2))
        units, meta = select_chapter_units(_tree(children))

        assert meta["unit_selection_tier"] == "chapter_vocabulary"
        assert _titles(units) == [
            "Chapter 1. Topic I-1",
            "Chapter 2. Topic I-2",
            "Chapter 3. Topic I-3",
            "Chapter 1. Topic II-1",
            "Chapter 2. Topic II-2",
            "Chapter 3. Topic II-3",
        ]

    def test_toc_cluster_and_answer_key_do_not_become_chapters(self):
        """CompTIA PenTest+: the TOC has no page numbers and the answer key
        repeats every chapter heading with a real amount of text each."""
        children = [_node("CompTIA PenTest+ Study Guide", "Front matter. " * 200)]
        children += [_node(f"Chapter {c}: Title {c}", "") for c in range(1, 7)]
        children.append(_node("Appendix: Answers to Review Questions", ""))
        for c in (1, 3, 5):
            children.append(_node(f"Chapter {c} Title {c}", BODY * 8, page=c * 50))
            children.append(_node(f"Body of chapter {c + 1}", BODY * 8, page=c * 50 + 25))
        children.append(_node("Appendix A: Answers to Review Questions", "", page=400))
        children += [_node(f"Chapter {c}: Title {c}", BODY * 2, page=400 + c) for c in range(1, 7)]
        units, meta = select_chapter_units(_tree(children))

        titles = _titles(units)
        assert meta["unit_selection_tier"] == "chapter_vocabulary"
        assert titles[-1] == "Appendix A: Answers to Review Questions"
        assert [t for t in titles if t.startswith("Chapter")] == [
            "Chapter 1 Title 1",
            "Chapter 3 Title 3",
            "Chapter 5 Title 5",
        ]

    def test_redundant_leading_number_still_matches(self):
        """Advanced Apple Debugging: «2 Chapter 2: Overview & Getting Help»."""
        children = [_node(f"{c} Chapter {c}: Topic {c}", BODY * 2) for c in range(1, 5)]
        units, meta = select_chapter_units(_tree(children))

        assert meta["unit_selection_tier"] == "chapter_vocabulary"
        assert len(units) == 4


class TestImplausibleHeadingsNeverTitleAUnit:
    JUNK = [
        "4) Стравливание окисла со всей поверхности.",
        "Обычно величина  \\( \\tau_{B} \\)  не превышает (2–5)",
        "9. Строковые команды (табл. 6.17)",
        "1 7 These and other circuit-related terms are explained in section 1.7.",
        "Содержание",
        "Контрольные вопросы",
        "In this case, we have,",
        "Indeed:",
    ]

    def test_root_children_skip_junk(self):
        children = []
        for i, junk in enumerate(self.JUNK):
            children.append(_node(f"Real Chapter Name {i}", BODY * 3))
            children.append(_node(junk, BODY * 3))
        units, _ = select_chapter_units(_tree(children))

        titles = _titles(units)
        for junk in self.JUNK:
            assert junk not in titles


class TestChapterNamesFromTheTableOfContents:
    """DOC_010 (Ru_Book): OCR never produced the «Глава k» lines, so every
    chapter was named after its first section, «k.1. Введение» — §2.2 printed
    "Giới thiệu" eight times. The book's own contents list has the names."""

    TOC = "\n".join(
        f"Глава {k}. {name}. {k * 30}\n{k}.1. Введение. {k * 30}\n{k}.2. Основы. {k * 30 + 5}"
        for k, name in enumerate(["Предмет микроэлектроники", "Полупроводники", "Транзисторы"], 1)
    )

    def _book(self):
        children = [_node("Содержание", self.TOC)]
        for k in range(1, 4):
            children.append(_node("Контрольные вопросы", "1. Вопрос? " * 20))
            for s in range(1, 5):
                children.append(_node(f"{k}.{s}. Введение" if s == 1 else f"{k}.{s}. Раздел", BODY))
        return _tree(children)

    def test_subsection_anchored_chapter_takes_its_contents_name(self):
        units, meta = select_chapter_units(self._book())

        assert meta["unit_selection_tier"] == "section_prefix"
        assert _titles(units) == [
            "Глава 1. Предмет микроэлектроники",
            "Глава 2. Полупроводники",
            "Глава 3. Транзисторы",
        ]


class TestChaptersClosedByEndMarkers:
    """DOC_012 (CompTIA PenTest+): no chapter heading survived in the body — only
    in the contents list and the answer key — so 12 chapters came out as 5. Each
    chapter still closes with Summary / Exam Essentials / Review Questions."""

    NAMES = ["Penetration Testing", "Planning and Scoping", "Information Gathering",
             "Vulnerability Scanning"]  # fmt: skip

    def _book(self, with_toc=True):
        children = [
            _node("Introduction", "How to use this book. " * 60),
            # The introduction describes the study aids by their headings.
            _node("Review Questions", "Each chapter ends with review questions. " * 5),
            _node("Summary", "Every chapter has a summary. " * 5),
        ]
        if with_toc:
            children += [_node(f"Chapter {k}: {name}") for k, name in enumerate(self.NAMES, 1)]
            children.append(_node("Appendix: Answers to Review Questions"))  # as in the book
        for k, name in enumerate(self.NAMES, 1):
            children.append(_node(f"Opening topic of {name.lower()}", BODY))
            children.append(_node(f"Deeper topic of {name.lower()}", BODY))
            children.append(_node("Summary", "In this chapter. " * 30))
            children.append(_node("Exam Essentials", "Know this. " * 30))
            children.append(_node("Review Questions", "1. Which of the following? " * 30))
        children.append(_node("Appendix: Answers to Review Questions", BODY))
        return _tree(children)

    def test_each_closing_block_ends_a_chapter_named_from_the_contents(self):
        units, meta = select_chapter_units(self._book())

        assert meta["unit_selection_tier"] == "end_markers"
        titles = _titles(units)
        chapters = [f"Chapter {k}: {n}" for k, n in enumerate(self.NAMES, 1)]
        assert titles[titles.index(chapters[0]) :][:4] == chapters
        assert titles[-1] == "Appendix: Answers to Review Questions"

    def test_a_review_question_never_names_the_unit_after_the_last_chapter(self):
        """DOC_012's answer-key unit came out as «Which of the following …?»."""
        tree = self._book()
        tree["children"].insert(-1, _node("1. Which of the following is true?", BODY))
        units, _ = select_chapter_units(tree)

        assert not any(t.endswith("?") for t in _titles(units))

    def test_back_matter_after_the_last_closing_is_named_by_its_heading(self):
        """DOC_012: a PowerShell line inside the last review question came before
        «Appendix A: Answers to Review Questions»."""
        tree = self._book()
        tree["children"].insert(-1, _node('Write-Host "The system contains serious flaws"', BODY))
        units, _ = select_chapter_units(tree)

        assert _titles(units)[-1] == "Appendix: Answers to Review Questions"

    def test_a_few_extra_closings_do_not_overrule_real_chapter_headings(self):
        """DOC_013: «Where to go from here?» closes 27 chapters but the body kept
        21 real «Chapter N» headings — the headings name chapters far better."""
        children = []
        for k in range(1, 9):
            if k not in (4, 7):
                children.append(_node(f"Chapter {k}: Topic {k}", ""))
            children.append(_node(f"First section of {k}", BODY))
            children.append(_node("Where to go from here?", "Next steps. " * 20))
        units, meta = select_chapter_units(_tree(children))

        assert meta["unit_selection_tier"] == "chapter_vocabulary"

    def test_a_front_matter_test_before_chapter_one_does_not_shift_the_contents_names(self):
        """DOC_024: an assessment test and a second description of the study aids sit
        between the contents list and chapter 1, so 14 closings met 12 listed chapters
        and no chapter took its contents name — they were named from wrong anchors."""
        tree = self._book()
        toc_end = next(i for i, n in enumerate(tree["children"]) if n["title"].startswith("Appendix"))
        front = [
            _node("Practice Test", BODY),
            _node("Real-World Scenarios", BODY),
            _node("Summary", "Each chapter has a summary. " * 30),
            _node("Exam Essentials", "Know this. " * 30),
            _node("Review Questions", "1. Which of the following? " * 30),
        ]
        tree["children"][toc_end + 1 : toc_end + 1] = front
        units, meta = select_chapter_units(tree)

        assert meta["unit_selection_tier"] == "end_markers"
        titles = _titles(units)
        chapters = [f"Chapter {k}: {n}" for k, n in enumerate(self.NAMES, 1)]
        assert titles[titles.index(chapters[0]) :][:4] == chapters
        assert "Practice Test" not in titles

    def test_a_real_appendix_heading_is_not_renamed_when_the_contents_name_the_chapters(self):
        units, _ = select_chapter_units(self._book())

        appendix = next(u for u in units if u["title"].startswith("Appendix"))
        assert not appendix.get("untitled")

    def test_without_a_contents_list_the_opening_heading_names_the_chapter(self):
        units, meta = select_chapter_units(self._book(with_toc=False))

        assert meta["unit_selection_tier"] == "end_markers"
        assert "Opening topic of information gathering" in _titles(units)


class TestJunkChapterNames:
    """The line before a chapter's first section named the chapter even when it
    was a sentence fragment (Ballistics: «The evaluation of which yields»), an
    exercise («Problem 2») or a quoted question (Digital IC: «3. During testing:
    “Can I tell intact circuits from defective ones?”»)."""

    JUNK = [
        "The evaluation of which yields",
        "The complex velocity can therefore be expressed as",
        "Problem 2",
        "3. During testing: “Can I tell intact circuits from defective ones?”",
    ]

    def test_junk_lines_never_name_a_chapter(self):
        children = []
        for k, junk in enumerate(self.JUNK, start=1):
            children.append(_node(junk, "Some text. " * 5))
            for s in range(1, 5):
                children.append(_node(f"{k}.{s} Section {s}", BODY))
        units, _ = select_chapter_units(_tree(children))

        assert not set(self.JUNK) & set(_titles(units))


def test_cp1251_mojibake_title_is_repaired():
    """Ru_Designing ch. 7: the text layer's «МИКРОПРОЦЕССОР» arrived as «ÌÈÊÐÎÏÐÎÖÅÑÑÎÐ»."""
    garbled = "МИКРОПРОЦЕССОР МОДЕЛИ 68000".encode("cp1251").decode("latin-1")
    children = []
    for k, name in enumerate(["Введение", garbled, "Интерфейсы"], start=1):
        children.append(_node(name, "Текст. " * 5))
        for s in range(1, 5):
            children.append(_node(f"{k}.{s} Раздел {s}", BODY))
    units, _ = select_chapter_units(_tree(children))

    assert "МИКРОПРОЦЕССОР МОДЕЛИ 68000" in _titles(units)


class TestSectionPrefixRecoversUnnumberedChapters:
    """Ballistics (502 pp., 19 chapters): §2.2 listed 14, because a chapter with
    no «N.1» section was folded into the one before it.

    - 7, 10, 19 have a bare heading (no numbered sections at all);
    - 6 and 14 open a Part and have no heading of their own: their text hangs
      under the Part's title page;
    - 11 and 16 first show a section («11.1», «16.2»), which became the title.

    The book's contents list names every chapter, so the gaps are filled from it.
    """

    @staticmethod
    def _book():
        import json
        from pathlib import Path

        path = Path(__file__).parents[2] / "fixtures" / "trees" / "ballistics_headings.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_every_chapter_gets_a_unit(self):
        units, meta = select_chapter_units(self._book())

        assert meta["unit_selection_tier"] == "section_prefix"
        assert [u.get("chapter_ordinal") for u in units] == list(range(1, 20))

    def test_units_are_named_after_their_chapter_not_a_section(self):
        units, _ = select_chapter_units(self._book())
        titles = _titles(units)

        assert not any(t.startswith(("11.1", "16.2")) for t in titles)
        for name in ("Dynamics Review", "Mass Asymmetries", "Lateral Throwoff", "Wound Ballistics"):
            assert any(name in t for t in titles), name

    def test_part_opening_chapters_are_not_named_after_the_part(self):
        units, _ = select_chapter_units(self._book())
        titles = _titles(units)

        assert sum("Introductory Concepts" in t for t in titles) == 3  # chapters 1, 6, 14


class TestUnitsKnowTheirPart:
    """§2.1 opens «Phần I … (chương 1 đến chương 5)»: it needs each chapter's Part."""

    def test_a_chapter_carries_the_part_it_sits_in(self):
        units, _ = select_chapter_units(TestSectionPrefixRecoversUnnumberedChapters._book())
        part_of = {u["chapter_ordinal"]: u.get("part") for u in units}

        assert part_of[1] == {"ordinal": 1, "name": "Interior Ballistics"}
        assert part_of[5] == {"ordinal": 1, "name": "Interior Ballistics"}
        assert part_of[6] == {"ordinal": 2, "name": "Exterior Ballistics"}
        assert part_of[13] == {"ordinal": 2, "name": "Exterior Ballistics"}
        assert part_of[14] == {"ordinal": 3, "name": "Terminal Ballistics"}
        assert part_of[19] == {"ordinal": 3, "name": "Terminal Ballistics"}

    def test_a_book_without_parts_has_none(self):
        units, _ = select_chapter_units(TestSectionPrefixTier()._book())

        assert all(u.get("part") is None for u in units)


class TestChapterOpenedByItsBareNumber:
    """Ballistics chapter 1 opens with a node titled just «1» (the number, then the
    first paragraph). §2.2 printed «Chương 1. 1.» — the contents list has its name."""

    TOC = "\n".join(
        f"Chapter {k} {name}\n{k}.1 Overview {k * 30}"
        for k, name in enumerate(["Introductory Concepts", "Gas Laws", "Combustion"], 1)
    )

    def _book(self):
        children = [_node("Contents", self.TOC)]
        for k in range(1, 4):
            if k == 1:
                children.append(_node("1", BODY, page=10))
            else:
                children.append(_node(["", "Gas Laws", "Combustion"][k - 1], "Intro. " * 5))
            for s in range(1, 5):
                children.append(_node(f"{k}.{s} Section {s}", BODY))
        return _tree(children)

    def test_a_number_only_heading_takes_the_contents_name(self):
        units, _ = select_chapter_units(self._book())

        assert _titles(units)[0] == "Chapter 1 Introductory Concepts"


class TestPapersInAVolume:
    """A kỷ yếu is many papers, each opening with its own abstract(s). The article tier
    anchors on each IMRaD heading's first occurrence, so a 13-paper volume (153 pp.)
    got «MỞ ĐẦU», «PHƯƠNG PHÁP», «KẾT QUẢ» … as its §2.2 entries instead of 13 papers."""

    @staticmethod
    def _volume(papers=4):
        children = []
        for k in range(papers):
            page = k * 12 + 1
            children.append(_node("TÓMTẮT", "", page=page))
            children.append(_node("ABSTRACT", "", page=page))
            for name in ("ĐẶT VẤN ĐỀ", "PHƯƠNG PHÁP NGHIÊN CỨU", "KẾT QUẢ", "KẾT LUẬN"):
                children.append(_node(name, BODY, page=page + 1))
        return _tree(children)

    def test_one_unit_per_paper(self):
        units, meta = select_chapter_units(self._volume(4))

        assert meta["unit_selection_tier"] == "paper_boundaries"
        assert len(units) == 4

    def test_a_paper_is_marked_as_needing_its_title(self):
        units, _ = select_chapter_units(self._volume(4))

        assert all(u.get("untitled") for u in units)

    def test_a_single_article_is_still_split_into_its_sections(self):
        units, meta = select_chapter_units(self._volume(1))

        assert meta["unit_selection_tier"] != "paper_boundaries"

    def test_an_english_abstract_trailing_the_last_paper_is_not_a_paper(self):
        """Live (153-page volume of 13 papers): the last paper's closing English abstract,
        2.3k characters, became a fourteenth entry."""
        volume = self._volume(4)
        volume["children"].append(_node("ABSTRACT", "Short English abstract. " * 8, page=60))

        units, _ = select_chapter_units(volume)

        assert len(units) == 4

    @staticmethod
    def _journal_volume(papers=4):
        """Each paper opens with the Vietnamese abstract and closes with the English one,
        so the English abstract of paper k sits right before the Vietnamese one of k+1,
        with the next paper's title in between (the real 13-paper volume)."""
        children = [_node("Tiêu đề bài 1", "", page=1)]
        for k in range(papers):
            page = k * 12 + 1
            children.append(_node("TÓMTẮT", "", page=page))
            for name in ("ĐẶT VẤN ĐỀ", "PHƯƠNG PHÁP NGHIÊN CỨU", "KẾT QUẢ", "KẾT LUẬN"):
                children.append(_node(name, BODY, page=page + 1))
            children.append(_node("ABSTRACT", "", page=page + 11))
            children.append(_node(f"Tiêu đề bài {k + 2}", "", page=page + 11))
        return _tree(children)

    def test_the_paper_opens_at_the_abstract_in_the_language_the_volume_starts_with(self):
        units, meta = select_chapter_units(self._journal_volume(4))

        assert meta["unit_selection_tier"] == "paper_boundaries"
        assert len(units) == 4
        # the English abstract closing paper k stays with paper k
        assert "ABSTRACT" in [c["title"] for c in units[0]["children"]]
        assert "ABSTRACT" not in [c["title"] for c in units[1]["children"][:2]]

    def test_the_title_printed_before_the_opening_abstract_is_offered_for_naming(self):
        units, _ = select_chapter_units(self._journal_volume(4))

        assert "Tiêu đề bài 2" in units[1]["title_hint"]


class TestPapersEndingAtTheirReferences:
    """English and Russian volumes (211 and 279 pages, 13 papers each) have no abstract
    heading to anchor on, but every paper ends with a reference list, in any language
    and template — sometimes twice (Литература, then References)."""

    @staticmethod
    def _volume(papers=4, bilingual_refs=False):
        children = []
        for k in range(papers):
            page = k * 10 + 1
            children.append(_node(f"Paper {k + 1} title", "", page=page))
            for name in ("Introduction", "Method", "Results", "Conclusion"):
                children.append(_node(f"{name} {k}", BODY, page=page + 1))
            children.append(_node("References", "1. Ref. " * 60, page=page + 8))
            if bilingual_refs:
                children.append(_node("Литература", "1. Ref. " * 60, page=page + 9))
        return _tree(children)

    def test_one_unit_per_reference_list(self):
        units, meta = select_chapter_units(self._volume(4))

        assert meta["unit_selection_tier"] == "paper_boundaries"
        assert len(units) == 4
        assert all(u.get("untitled") for u in units)

    def test_each_paper_starts_right_after_the_previous_reference_list(self):
        units, _ = select_chapter_units(self._volume(4))

        assert "Paper 2 title" in units[1]["title_hint"]
        assert "Paper 2 title" not in units[0]["title_hint"]

    def test_two_reference_lists_in_a_row_are_one_paper(self):
        units, _ = select_chapter_units(self._volume(4, bilingual_refs=True))

        assert len(units) == 4

    def test_wordings_seen_in_real_volumes_are_reference_headings(self):
        from utils.chapter_units import _is_reference_heading

        for title in (
            "References:",
            "6 References",
            "СПИСОК ЛИТЕРАТУРЫ / REFERENCES",
            "Список литературы на русском языке:",
            "Литература",
            "TÀI LIỆU THAM KHẢO",
        ):
            assert _is_reference_heading(title), title
        for title in ("Обзор литературы", "Lack of Resources", "Источник: рассчитано автором"):
            assert not _is_reference_heading(title), title


class TestArticleFrontMatterAndReferencesAreNotChapters:
    """E2E 2026-10 (DOC_029, DOC_031): a 5-page Russian article was split into
    «УДК 534.87» and «Литература» — the classification code and the reference list.
    The trees are the real ones."""

    @staticmethod
    def _fixture(name):
        import json
        from pathlib import Path

        path = Path(__file__).parents[2] / "fixtures" / "trees" / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_classification_code_and_the_reference_list_are_not_units(self):
        for name in ("article_udk_ru", "article_udk_ru_2", "article_two_methods_ru"):
            units, _ = select_chapter_units(self._fixture(name))

            titles = [t.casefold() for t in _titles(units)]
            assert units, name
            assert not any(t.startswith("удк") or t == "литература" for t in titles), (
                name,
                titles,
            )


class TestUnitsWhoseTitleIsNotAName:
    """E2E 2026-10: the chapter list printed «Tài liệu tham khảo» five times
    (DOC_020), «Đáp án nằm ở Phụ lục A» (DOC_024), two blank names for «C H A P T E R 2»
    (DOC_017) and a reference entry for an article section (DOC_029). Where the tree
    holds no name, the unit is marked so the model names it from its own text."""

    def test_what_is_not_a_heading_needs_a_name(self):
        from utils.chapter_units import _title_needs_naming

        for title in (
            "You can find the answers in the Appendix A.",
            "Resources",
            "Summary",
            "C H A P T E R 2",
            "1. Демидов Г.А. Основы теории горения и взрыва",
            "Phần 3",
        ):
            assert _title_needs_naming(title), title

    def test_a_running_page_header_repeating_the_book_title_needs_a_name(self):
        """DOC_017: «2 DIGITAL CONTROL IN POWER ELECTRONICS» is the page header, not chapter 1."""
        from utils.chapter_units import _title_needs_naming

        running = ("digital control in power electronics",)
        assert _title_needs_naming("2 DIGITAL CONTROL IN POWER ELECTRONICS", running)
        assert not _title_needs_naming("2 Introduction", running)

    def test_a_real_heading_keeps_its_name(self):
        from utils.chapter_units import _title_needs_naming

        for title in (
            "Information Gathering",
            "Chapter 3: Information Gathering",
            "Basic Topologies",
            "Глава 2. Полупроводники",
            "Preface",  # the tree's own label for front matter; real books have one
            "9 INTERNATIONAL GOVERNANCE, SOCIETAL ACCOUNTABILITY AND RISK MANAGEMENT",
        ):
            assert not _title_needs_naming(title), title

    def test_units_of_a_headingless_article_are_all_marked_with_their_opening(self):
        import json
        from pathlib import Path

        path = Path(__file__).parents[2] / "fixtures" / "trees" / "article_udk_ru_2.json"
        units, meta = select_chapter_units(json.loads(path.read_text(encoding="utf-8")))

        assert meta["unit_selection_tier"] == "mass_segmentation"
        assert all(u.get("untitled") and u.get("title_hint") for u in units)
        assert not any(u.get("single_paper") for u in units)

    def test_a_book_with_real_chapter_headings_is_not_marked(self):
        tree = _tree(
            [
                _node(f"Chapter {k} Topic {k}", BODY, page=k * 10)
                for k in range(1, 5)
            ]
        )
        units, _ = select_chapter_units(tree)

        assert units and not any(u.get("untitled") for u in units)
