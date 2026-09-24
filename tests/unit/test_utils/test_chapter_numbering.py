"""§2.2 chapter labels, from the rendered digests of the 2026-09 E2E books.

- The "Các phần bổ trợ" entry was printed as "Chương 1", so every real chapter
  of DOC_002/003/004/006/007/009 came out one number too high.
- Entries without a recognised label took their list position and mixed with
  the book's own numbers (DOC_013: 11 → 17 → 13).
- Titles kept their source prefix ("2 Chapter 2: …", "2. What is…") and the
  text layer's tabs (DOC_014).
"""

from utils.chapter_numbering import AUX_TITLE_ORIGINAL, normalize_chapter_entries
from utils.digest_format import chapter_heading


def _aux():
    return {"number": 1, "title_vi": "Các phần bổ trợ", "title_original": AUX_TITLE_ORIGINAL}


def _ch(number, original, vi=None, **extra):
    return {"number": number, "title_vi": vi or f"VI {number}", "title_original": original, **extra}


def _labels(entries, doc_kind="book"):
    return [
        chapter_heading(
            e["number"],
            e["title_vi"],
            e["title_original"],
            doc_kind=doc_kind,
            heading_kind=e.get("heading_kind"),
            heading_ordinal=e.get("heading_ordinal"),
        ).split(".")[0]
        for e in entries
    ]


def test_auxiliary_entry_takes_no_chapter_number():
    entries = normalize_chapter_entries(
        [_aux(), _ch(2, "Basic Topologies"), _ch(3, "Push-Pull Topologies")]
    )
    # The English "original" is our own placeholder, not the book's — not printed.
    assert _labels(entries) == ["Các phần bổ trợ", "Chương 1", "Chương 2"]


def test_auxiliary_entry_is_not_a_paper_in_proceedings():
    entries = normalize_chapter_entries([_aux(), _ch(2, "Paper one")])
    assert not _labels(entries, "proceedings")[0].startswith("BBKH")
    assert _labels(entries, "proceedings")[1] == "BBKH 1 - VI 2 (Paper one)"


def test_docling_numbered_label_is_parsed_and_stripped():
    [entry] = normalize_chapter_entries([_ch(5, "6 Chapter 6: Thread, Frame & Stepping")])
    assert entry["title_original"] == "Thread, Frame & Stepping"
    assert (entry["heading_kind"], entry["heading_ordinal"]) == ("chapter", 6)


def test_unlabelled_chapter_follows_the_previous_chapter_number():
    entries = normalize_chapter_entries(
        [
            _aux(),
            _ch(11, "16 Chapter 16: Hooking"),
            _ch(12, "Hello, Mach-O", heading_kind="chapter", heading_ordinal=17),
            _ch(13, "Untitled but real"),
            _ch(14, "20 Chapter 20: Hello, Script Bridging"),
        ]
    )
    assert _labels(entries)[1:] == ["Chương 16", "Chương 17", "Chương 18", "Chương 20"]


def test_consistent_leading_numbers_are_the_chapter_numbers():
    """DOC_008: "2. What is a Digital Twin?" … "10. Case Studies" printed as Chương 1…9."""
    entries = normalize_chapter_entries(
        [_ch(1, "2. What is a Digital Twin?"), _ch(2, "3. Architecture"), _ch(3, "4. Types")]
    )
    assert _labels(entries) == ["Chương 2", "Chương 3", "Chương 4"]
    assert entries[0]["title_original"] == "What is a Digital Twin?"


def test_a_lone_leading_number_is_left_alone():
    """One "3. During testing…" among unnumbered titles is not a numbering scheme."""
    entries = normalize_chapter_entries(
        [_ch(1, "From Algorithms"), _ch(2, "3. During testing"), _ch(3, "Modelling Hardware")]
    )
    assert _labels(entries) == ["Chương 1", "Chương 2", "Chương 3"]
    assert entries[1]["title_original"] == "3. During testing"


def test_whitespace_in_titles_is_collapsed():
    [entry] = normalize_chapter_entries([_ch(1, "Sharpening\tthe\tSaw", vi="Rèn  luyện")])
    assert entry["title_original"] == "Sharpening the Saw"
    assert entry["title_vi"] == "Rèn luyện"


def test_appendix_keeps_its_letter_and_does_not_move_the_chapter_count():
    entries = normalize_chapter_entries(
        [
            _ch(1, "Penetration Testing"),
            _ch(2, "Answers", heading_kind="appendix", heading_ordinal=1),
            _ch(3, "Scanning"),
        ]
    )
    assert _labels(entries) == ["Chương 1", "Phụ lục A", "Chương 2"]


def test_input_is_not_mutated():
    raw = [_aux(), _ch(2, "2 Chapter 2: Overview")]
    normalize_chapter_entries(raw)
    assert raw[1]["title_original"] == "2 Chapter 2: Overview"


def test_unnumbered_unit_colliding_with_the_next_chapter_prints_no_number():
    """DOC_012: «Chương 1. Lời nói đầu» then «Chương 1. Kiểm thử xâm nhập»."""
    entries = normalize_chapter_entries(
        [
            _ch(1, "Preface", vi="Lời nói đầu"),
            _ch(2, "Chapter 1: Penetration Testing"),
            _ch(3, "Chapter 2: Planning"),
        ]
    )
    assert _labels(entries) == ["Lời nói đầu (Preface)", "Chương 1", "Chương 2"]


def test_original_title_is_not_repeated_inside_the_vietnamese_one():
    """DOC_003: «Đạn công phá (Shaped Charges) (Shaped Charges)»."""
    [entry] = normalize_chapter_entries(
        [_ch(1, "Shaped Charges", vi="Đạn công phá (Shaped Charges)")]
    )
    assert entry["title_vi"] == "Đạn công phá"
