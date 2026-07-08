#!/usr/bin/env python3
"""Generate docuflow_digest_template.docx with docxtpl Jinja2 placeholders."""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "template" / "docuflow_digest_template.docx"


def _sep(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("*****")
    r.bold = True


def main():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    h1 = doc.add_paragraph()
    h1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = h1.add_run("HỌC VIỆN KỸ THUẬT QUÂN SỰ")
    r.bold = True
    r.font.size = Pt(13)

    h2 = doc.add_paragraph()
    h2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = h2.add_run("PHÒNG THÔNG TIN KHOA HỌC QUÂN SỰ")
    r2.bold = True

    _sep(doc)

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rt = t.add_run("TỔNG THUẬT TÀI LIỆU")
    rt.bold = True
    rt.font.size = Pt(14)

    doc.add_paragraph()

    # §1
    p = doc.add_paragraph()
    p.add_run("1. THÔNG TIN CHUNG VỀ TÀI LIỆU").bold = True

    for label, key in [
        ("Tên tài liệu (Title)", "title_display"),
        ("Tác giả (Authors)", "authors"),
        ("Nhà xuất bản (Publisher)", "publisher"),
        ("Năm xuất bản (Year)", "year"),
        ("ISBN (ISBN)", "isbn"),
        ("DOI", "doi"),
        ("Số trang (Pages)", "pages"),
    ]:
        fp = doc.add_paragraph()
        fp.add_run(f"- {label}: ").bold = True
        fp.add_run(f"{{{{ bib.{key} }}}}")

    doc.add_paragraph()

    # §2
    p2 = doc.add_paragraph()
    p2.add_run("2. TỔNG THUẬT VỀ TÀI LIỆU").bold = True

    s21 = doc.add_paragraph()
    s21.add_run("2.1. Tóm Tắt").bold = True
    doc.add_paragraph("{{ abstract }}")

    doc.add_paragraph()
    s22 = doc.add_paragraph()
    s22.add_run("2.2. Nội dung chính của tài liệu").bold = True
    doc.add_paragraph("{% for ch in chapters %}")
    doc.add_paragraph(
        "{{ ch.number }}. {{ ch.title_vi }} ({{ ch.title_original }}). {{ ch.content }}"
    )
    doc.add_paragraph("{% endfor %}")
    doc.add_paragraph("{% if not chapters %}[Chưa có — chạy main_content trước]{% endif %}")

    doc.add_paragraph()
    s23 = doc.add_paragraph()
    s23.add_run("2.3. Từ Khóa").bold = True
    doc.add_paragraph("{% for kw in keywords %}")
    doc.add_paragraph("- {{ kw.display }}")
    doc.add_paragraph("{% endfor %}")

    doc.add_paragraph()

    # §3
    p3 = doc.add_paragraph()
    p3.add_run("3. Phạm vi sử dụng").bold = True

    for label, key in [
        ("CTĐT đại học", "undergraduate_text"),
        ("CTĐT thạc sĩ", "master_text"),
        ("CTĐT tiến sĩ", "phd_text"),
        ("Nhóm nghiên cứu mạnh", "strong_research_groups_text"),
    ]:
        fp = doc.add_paragraph()
        fp.add_run(f"- {label}: ").bold = True
        fp.add_run(f"{{{{ usage.{key} }}}}")

    rd = doc.add_paragraph()
    rd.add_run("- Hướng nghiên cứu: ").bold = True
    doc.add_paragraph("{% for rd in research_directions %}")
    doc.add_paragraph("- {{ rd.direction_name }}")
    doc.add_paragraph("{% endfor %}")

    doc.add_paragraph()
    _sep(doc)

    admin = doc.add_paragraph()
    admin.add_run("* Thông tin quản trị CSDL:").bold = True
    for label, key in [
        ("Người tổng thuật", "reviewer"),
        ("Người hiệu đính, phê duyệt", "reviewer_approved"),
        ("Ngày nhập CSDL Học liệu số", "entry_date"),
    ]:
        fp = doc.add_paragraph()
        fp.add_run(f"- {label}: ").bold = True
        fp.add_run(f"{{{{ {key} }}}}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
