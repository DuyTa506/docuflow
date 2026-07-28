"""§2.2 phải đổi dạng khi tài liệu là kỷ yếu — kiểm bằng cách đọc lại .docx.

Test dạng dòng nằm ở tests/unit/test_utils/test_digest_heading.py; ở đây chỉ
kiểm hai thứ mà hàm thuần không kiểm được: dòng đó có thật sự đi qua template
hay không, và chế độ Book có bị ảnh hưởng hay không.
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


def _digest(*, doc_kind="book", chapters=None):
    return DigestResult(
        document_id="DOC_001",
        title="Kỷ yếu HNKH 2025",
        source_language="vi",
        original_filename="ky-yeu.pdf",
        bibliographic={"title_display": "Kỷ yếu HNKH 2025", "pages": "640"},
        abstract="Tài liệu là tuyển tập 101 BBKH, chia thành 13 phần.",
        doc_kind=doc_kind,
        chapters=chapters or [],
        keywords=[KeywordEntry(keyword="k", display="Từ khoá (k)", weight=0.9)],
        usage_scope={"undergraduate": [], "master": [], "phd": [], "strong_research_groups": []},
        research_directions=[ResearchDirectionEntry(direction_name="Hướng 1", confidence=0.8)],
    )


def _text(digest) -> str:
    data = DigestRenderer().render(digest)
    return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)


class TestProceedingsRendering:
    def test_clusters_render_with_the_paper_count(self):
        digest = _digest(
            doc_kind="proceedings",
            chapters=[
                ChapterEntry(
                    number=1,
                    title_vi="Khoa học máy tính",
                    title_original="Computer Science",
                    content="Các nghiên cứu về phân loại botnet trên Android.",
                    paper_count=5,
                )
            ],
        )

        assert (
            "Khoa học máy tính (Computer Science), gồm 5 BBKH. "
            "Các nghiên cứu về phân loại botnet trên Android." in _text(digest)
        )

    def test_individual_papers_render_as_bbkh(self):
        digest = _digest(
            doc_kind="proceedings",
            chapters=[
                ChapterEntry(
                    number=2,
                    title_vi="Giám sát chất lượng nước",
                    title_original="Real-Time Water Quality Monitoring",
                    content="Nghiên cứu công nghệ cảm biến hoá học.",
                )
            ],
        )

        text = _text(digest)

        assert "BBKH 2 - Giám sát chất lượng nước (Real-Time Water Quality Monitoring)." in text
        assert "Chương 2." not in text


class TestBookModeUnchanged:
    def test_book_still_uses_the_chuong_prefix(self):
        digest = _digest(
            chapters=[
                ChapterEntry(
                    number=1,
                    title_vi="Giới thiệu",
                    title_original="Introduction",
                    content="Bối cảnh nghiên cứu.",
                )
            ]
        )

        assert "Chương 1. Giới thiệu (Introduction). Bối cảnh nghiên cứu." in _text(digest)

    def test_doc_kind_defaults_to_book(self):
        assert (
            DigestResult(
                document_id="DOC_002",
                title="x",
                source_language="vi",
                original_filename=None,
            ).doc_kind
            == "book"
        )
