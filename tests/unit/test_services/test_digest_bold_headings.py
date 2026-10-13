"""§2.2 của mẫu: nhãn chương nổi bật, nội dung bình thường, vẫn cùng MỘT đoạn.

Nhãn do code ghép (`chapter_heading`), nên in đậm ở tầng render chứ không nhờ model
sinh `**…**` — model đổi là hành vi đổi.
"""

import io

from docx import Document

from services.digest_renderer import DigestRenderer
from services.digest_service import (
    ChapterEntry,
    DigestResult,
    KeywordEntry,
    ResearchDirectionEntry,
)


def _digest(title_vi="Giới thiệu", content="Nội dung chương.", doc_kind="book", number=1):
    return DigestResult(
        document_id="DOC_001",
        title="T",
        source_language="ru",
        original_filename=None,
        bibliographic={"title_display": "T", "pages": "1"},
        abstract="Tóm tắt.",
        chapters=[
            ChapterEntry(
                number=number,
                title_vi=title_vi,
                title_original="Введение",
                content=content,
                heading_kind="chapter",
                heading_ordinal=number,
            )
        ],
        keywords=[KeywordEntry(keyword="k", display="Từ khoá (k)", weight=0.9)],
        usage_scope={"undergraduate": [], "master": [], "phd": [], "strong_research_groups": []},
        research_directions=[ResearchDirectionEntry(direction_name="H", confidence=0.8)],
        doc_kind=doc_kind,
    )


def _entry(digest, startswith):
    doc = Document(io.BytesIO(DigestRenderer().render(digest)))
    found = [p for p in doc.paragraphs if p.text.startswith(startswith)]
    assert len(found) == 1, [p.text[:40] for p in doc.paragraphs]
    return found[0]


def _split(paragraph):
    bold = "".join(r.text for r in paragraph.runs if r.bold)
    plain = "".join(r.text for r in paragraph.runs if not r.bold)
    return bold, plain


def test_the_chapter_label_is_bold_and_the_content_is_not():
    bold, plain = _split(_entry(_digest(), "Chương 1."))

    assert bold.strip() == "Chương 1. Giới thiệu (Введение)."
    assert "Nội dung chương." in plain
    assert "Chương" not in plain


def test_label_and_content_stay_in_one_paragraph():
    paragraph = _entry(_digest(), "Chương 1.")

    assert "Giới thiệu (Введение). Nội dung chương." in paragraph.text


def test_a_label_with_markdown_characters_is_printed_as_written_and_bold():
    bold, _ = _split(_entry(_digest(title_vi="Ký hiệu *p* và a_b"), "Chương 1."))

    assert "Ký hiệu *p* và a_b" in bold


def test_a_paper_label_of_a_proceedings_is_bold_too():
    digest = _digest(title_vi="Giáo dục STEM", doc_kind="proceedings", number=2)
    bold, plain = _split(_entry(digest, "BBKH 2"))

    assert bold.strip().startswith("BBKH 2")
    assert "Nội dung chương." in plain
