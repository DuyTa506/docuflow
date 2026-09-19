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
