"""The mẫu's footnote: «Book».«Số cá biệt».«Nhóm»_«Người tổng thuật»_«Ngày nhập CSDL»,
e.g. `B.000001.N1_VKSon_29.12.2024`; for books still being summarised «tạm lấy theo mã file»."""

import io

from docx import Document

from services.digest_renderer import DigestRenderer
from services.digest_service import DigestResult
from utils.digest_admin import normalize_digest_admin
from utils.digest_format import format_footer_code


class TestFormat:
    def test_the_mau_example(self):
        code = format_footer_code("000001", "N1", "VKSon", "29/12/2024", "DOC_001")

        assert code == "B.000001.N1_VKSon_29.12.2024"

    def test_without_a_catalogue_number_the_file_code_stands_in(self):
        assert (
            format_footer_code("", "N1", "VKSon", "29.12.2024", "DOC_018")
            == "B.DOC_018.N1_VKSon_29.12.2024"
        )

    def test_missing_trailing_pieces_leave_no_dangling_separators(self):
        assert format_footer_code("000007", "", "", "", "DOC_007") == "B.000007"
        assert format_footer_code("000007", "N2", "", "", "DOC_007") == "B.000007.N2"


class TestAdminKeys:
    def test_the_block_keeps_the_footer_fields(self):
        cleaned = normalize_digest_admin(
            {"catalog_number": " 000001 ", "group": "N1", "summarizer_code": "VKSon"}
        )

        assert cleaned["catalog_number"] == "000001" and cleaned["group"] == "N1"
        assert cleaned["summarizer_code"] == "VKSon"


def test_the_rendered_page_carries_the_footer_right_aligned():
    digest = DigestResult(
        document_id="DOC_001",
        title="T",
        source_language="en",
        original_filename=None,
        bibliographic={"title_display": "T"},
        catalog_number="000001",
        group="N1",
        summarizer_code="VKSon",
        entry_date="29/12/2024",
    )

    doc = Document(io.BytesIO(DigestRenderer().render(digest)))
    footer = doc.sections[0].footer.paragraphs[0]

    assert footer.text == "B.000001.N1_VKSon_29.12.2024"
    assert footer.alignment == 2  # WD_ALIGN_PARAGRAPH.RIGHT
