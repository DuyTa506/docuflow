"""Khung mục đúng, nội dung bên trong từng mục thì lệch mẫu.

The existing template test only checked that the bytes start with "PK" and are
over 1 kB, so every one of these drifted without a single failing test:
markdown printed as literal `##`, a blank paragraph per loop iteration, the
`Chương N.` prefix missing, research directions as a bullet list where the
official form is inline, heading casing, and an admin block that could never
be filled in.

Everything here reads the rendered .docx back and asserts on its text.
"""

import io

import pytest
from docx import Document

from services.digest_renderer import DigestRenderer
from services.digest_service import (
    ChapterEntry,
    DigestResult,
    KeywordEntry,
    ResearchDirectionEntry,
)


def _digest(*, chapters=1, keywords=1, directions=1, abstract="Tóm tắt thử nghiệm.", **kw):
    return DigestResult(
        document_id="DOC_001",
        title="Test Book",
        source_language="ru",
        original_filename="test.pdf",
        bibliographic={"title_display": "Kiến trúc máy tính", "pages": "816"},
        abstract=abstract,
        chapters=[
            ChapterEntry(
                number=i + 1,
                title_vi=f"Chủ đề {i + 1}",
                title_original=f"Тема {i + 1}",
                content=f"Nội dung chương {i + 1}.",
            )
            for i in range(chapters)
        ],
        keywords=[
            KeywordEntry(keyword=f"term{i}", display=f"Thuật ngữ {i} (term{i})", weight=0.9)
            for i in range(keywords)
        ],
        usage_scope={
            "undergraduate": ["Ngành Khoa học máy tính"],
            "master": [],
            "phd": [],
            "strong_research_groups": [],
        },
        research_directions=[
            ResearchDirectionEntry(direction_name=f"Hướng {i + 1}", confidence=0.8)
            for i in range(directions)
        ],
        **kw,
    )


def _paragraphs(digest) -> list[str]:
    data = DigestRenderer().render(digest)
    return [p.text for p in Document(io.BytesIO(data)).paragraphs]


def _text(digest) -> str:
    return "\n".join(_paragraphs(digest))


class TestHeadingCasing:
    """Mẫu chính thức: `3. PHẠM VI SỬ DỤNG`, `2.1. Tóm tắt`, `2.3. Từ khóa`."""

    @pytest.mark.parametrize(
        "heading",
        [
            "1. THÔNG TIN CHUNG VỀ TÀI LIỆU",
            "2. TỔNG THUẬT VỀ TÀI LIỆU",
            "2.1. Tóm tắt",
            "2.2. Nội dung chính của tài liệu",
            "2.3. Từ khóa",
            "3. PHẠM VI SỬ DỤNG",
        ],
    )
    def test_heading_matches_the_official_form(self, heading):
        assert heading in _paragraphs(_digest())


class TestSectionOneLabels:
    """Nhãn §1 phải khớp `Mau Tong thuat Book .docx`, từng chữ.

    Ba nhãn đã trôi khỏi mẫu: "Tên tài liệu (Title)" thay vì "Nhan đề (Title)",
    và "(ISBN)" / "(Pages)" thừa ra sau hai nhãn mà mẫu để trần.
    """

    @pytest.mark.parametrize(
        "label",
        [
            "- Nhan đề (Title): ",
            "- Tác giả (Authors): ",
            "- Nhà xuất bản (Publisher): ",
            "- Năm xuất bản (Year): ",
            "- ISBN: ",
            "- DOI: ",
            "- Số trang: ",
        ],
    )
    def test_label_matches_the_official_form(self, label):
        paragraphs = _paragraphs(_digest())

        assert any(p.startswith(label) for p in paragraphs), f"thiếu nhãn «{label.strip()}»"


class TestChapterLine:
    def test_chapter_number_carries_the_chuong_prefix(self):
        """Mẫu: `Chương 1. Giới thiệu (Введение).`"""
        text = _text(_digest(chapters=2))

        assert "Chương 1. Chủ đề 1 (Тема 1)." in text
        assert "Chương 2. Chủ đề 2 (Тема 2)." in text


class TestResearchDirectionsInline:
    def test_directions_are_joined_inline_like_the_other_bullets(self):
        """§3 has four `- Nhãn: a; b; c` bullets; this was the odd one out."""
        paragraphs = _paragraphs(_digest(directions=3))

        line = next(p for p in paragraphs if p.startswith("- Hướng nghiên cứu:"))
        assert line == "- Hướng nghiên cứu: Hướng 1; Hướng 2; Hướng 3"

    def test_no_direction_leaves_an_empty_label_not_a_stray_bullet(self):
        paragraphs = _paragraphs(_digest(directions=0))

        assert "- Hướng nghiên cứu: " in paragraphs


class TestNoBlankLineInflation:
    def test_loops_do_not_add_a_blank_paragraph_per_iteration(self):
        """`{% for %}` on its own paragraph leaves that paragraph behind."""
        small = _paragraphs(_digest(chapters=1, keywords=1, directions=1))
        large = _paragraphs(_digest(chapters=6, keywords=6, directions=6))

        blanks = lambda ps: sum(1 for p in ps if not p.strip())  # noqa: E731
        assert blanks(small) == blanks(large)


class TestAbstractRendering:
    def test_markdown_markers_are_not_printed_literally(self):
        abstract = "## Chủ đề chính\n\nTài liệu **quan trọng** trình bày kiến trúc."

        paragraphs = _paragraphs(_digest(abstract=abstract))

        assert "Chủ đề chính" in paragraphs
        assert "Tài liệu quan trọng trình bày kiến trúc." in paragraphs
        # The `*****` masthead separator is legitimately made of asterisks, so
        # look at the abstract's own paragraphs, not the whole document.
        assert not any(
            "#" in p or "**" in p for p in paragraphs if "Chủ đề" in p or "Tài liệu" in p
        )

    def test_paragraph_breaks_survive_as_separate_paragraphs(self):
        abstract = "Đoạn thứ nhất của tóm tắt.\n\nĐoạn thứ hai của tóm tắt."

        paragraphs = _paragraphs(_digest(abstract=abstract))

        assert "Đoạn thứ nhất của tóm tắt." in paragraphs
        assert "Đoạn thứ hai của tóm tắt." in paragraphs

    def test_missing_abstract_still_says_so(self):
        assert "[Chưa có — chạy summarize trước]" in _text(_digest(abstract=None))

    def test_chapter_content_markdown_is_cleaned_too(self):
        digest = _digest()
        digest.chapters[0].content = "Chương này **nhấn mạnh** kiến trúc."

        assert "Chương này nhấn mạnh kiến trúc." in _text(digest)


class TestAdminBlock:
    def test_admin_fields_reach_the_document(self):
        digest = _digest(
            reviewer="Nguyễn Văn A",
            reviewer_approved="Trần Văn B",
            entry_date="28/07/2026",
        )

        text = _text(digest)

        assert "- Người tổng thuật: Nguyễn Văn A" in text
        assert "- Người hiệu đính, phê duyệt: Trần Văn B" in text
        assert "- Ngày nhập CSDL Học liệu số: 28/07/2026" in text
