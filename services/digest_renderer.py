"""
Digest DOCX renderer — docxtpl template matching Mau Tong thuat Book.
"""

import io
from pathlib import Path
from typing import Optional

from docxtpl import DocxTemplate

from services.digest_service import DIGEST_KEYWORD_TARGET, DigestResult
from utils.digest_format import (
    chapter_heading,
    collapse_bilingual_display,
    correct_unit_kind_words,
    drop_bibliographic_keywords,
    drop_heading_keywords,
    join_catalog_items,
    open_with_substance,
    plain_text,
    split_block_lines,
    strip_block_markdown,
)

_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "template" / "docuflow_digest_template.docx"
)


class DigestRenderer:
    """Render DigestResult to .docx via docxtpl."""

    def __init__(self, template_path: Optional[Path] = None):
        self.template_path = template_path or _TEMPLATE_PATH

    @staticmethod
    def _chapter_body(digest, c) -> str:
        """The complete §2.2 line: heading plus a body that names its unit correctly.

        The heading already prints "Phụ lục A. Số nhị phân (Двоичные числа).", so
        a body opening with "Phụ lục A (Приложение А) trình bày…" says it twice —
        8 of the 12 entries in N4.11.160 did. The prompt now forbids restating it;
        this is the backstop for when the model ignores that.

        The same entries then called themselves "Chương này" further down, so an
        appendix was described as a chapter in an official document; the kind is
        known here, so that self-reference is corrected too.
        """
        heading = chapter_heading(
            c.number,
            c.title_vi,
            c.title_original,
            doc_kind=digest.doc_kind,
            paper_count=c.paper_count,
            heading_kind=c.heading_kind,
            heading_ordinal=c.heading_ordinal,
        )
        label = heading.split(".", 1)[0]
        body = correct_unit_kind_words(open_with_substance(c.content, label), c.heading_kind)
        return f"{heading} {body}".strip()

    @staticmethod
    def _rich_paragraph(tpl: DocxTemplate, text: str):
        """One Word paragraph carrying real formatting, not literal markup.

        `_add_inline_runs` is the same renderer the text-download path uses: it
        splits `$...$` / `$$...$$`, checks each with `looks_like_math` (so `$100`
        stays a price) and inserts native OMML equations, and turns `**bold**`
        into a bold run instead of printing the asterisks.

        The digest used to flatten everything with `plain_text()`, so a chapter
        summary mentioning `$\\Delta w$` printed the dollars and the backslash
        into the official document. Fixing the renderer rather than forbidding
        the model is what survives a model swap — Gemma writes LaTeX where Qwen
        did not.
        """
        from utils.markdown_docx import _add_inline_runs

        sub = tpl.new_subdoc()
        para = sub.add_paragraph()
        # strip_html=False: the summary may be *about* a tag. python-docx writes
        # run text through the XML serialiser, so `<a>` is escaped safely — the
        # DOC_066 corruption came from docxtpl substituting into raw XML, which
        # this path no longer does.
        _add_inline_runs(para, strip_block_markdown(text), strip_html=False)
        return sub

    def _build_context(
        self,
        digest: DigestResult,
        tpl: DocxTemplate,
        reviewer: str = "",
        reviewer_approved: str = "",
        entry_date: str = "",
    ) -> dict:
        usage = digest.usage_scope or {}
        context_bib = digest.bibliographic or {}
        abstract_lines = split_block_lines(digest.abstract) or ["[Chưa có — chạy summarize trước]"]
        return {
            "bib": context_bib,
            # One paragraph per line: Word drops newlines inside a run, so a
            # single placeholder rendered the whole abstract as one block.
            "abstract_paragraphs": [self._rich_paragraph(tpl, line) for line in abstract_lines],
            # `heading` is composed here rather than in the template: the
            # official form differs by document kind (Chương N. / cụm BBKH /
            # BBKH N -) and three Jinja branches inside a Word paragraph is
            # far harder to read — and to test — than one pure function.
            #
            # Heading and content share ONE paragraph, as the mẫu requires
            # (`Chương 1. Giới thiệu (Введение). Nội dung…`), so they are built
            # into a single subdoc rather than two template placeholders.
            "chapters": [
                {
                    "number": c.number,
                    "title_vi": plain_text(c.title_vi),
                    "title_original": plain_text(c.title_original),
                    "body": self._rich_paragraph(
                        tpl,
                        self._chapter_body(digest, c),
                    ),
                }
                for c in digest.chapters
            ],
            # Title, author names (§1) and chapter headings (§2.2) are all
            # printed above. The keywords stage runs in parallel, so assembly is
            # the only place that knows all three and can filter.
            #
            # `digest.keywords` is a ranked pool larger than the target, so the
            # truncation happens *after* filtering: a rejected keyword costs a
            # slot otherwise, which is how §2.3 came out at 13 of 20. Slicing a
            # short list is still short — nothing is padded to reach the target.
            "keywords": [
                {
                    "display": collapse_bilingual_display(k["display"] or k["keyword"]),
                    "keyword": k["keyword"],
                    "weight": k["weight"],
                }
                for k in drop_heading_keywords(
                    drop_bibliographic_keywords(
                        [
                            {"display": k.display, "keyword": k.keyword, "weight": k.weight}
                            for k in digest.keywords
                        ],
                        context_bib,
                    ),
                    [t for c in digest.chapters for t in (c.title_vi, c.title_original)],
                )[:DIGEST_KEYWORD_TARGET]
            ],
            "usage": {
                "undergraduate_text": join_catalog_items(usage.get("undergraduate", [])),
                "master_text": join_catalog_items(usage.get("master", [])),
                "phd_text": join_catalog_items(usage.get("phd", [])),
                "strong_research_groups_text": join_catalog_items(
                    usage.get("strong_research_groups", [])
                ),
            },
            # §3's other four bullets are `- Nhãn: a; b; c`; this one used to be
            # an empty label followed by a separate bullet list.
            "research_directions_text": join_catalog_items(
                [rd.direction_name for rd in digest.research_directions]
            ),
            "reviewer": reviewer or digest.reviewer,
            "reviewer_approved": reviewer_approved or digest.reviewer_approved,
            "entry_date": entry_date or digest.entry_date,
        }

    def render(
        self,
        digest: DigestResult,
        reviewer: str = "",
        reviewer_approved: str = "",
        entry_date: str = "",
    ) -> bytes:
        if not self.template_path.is_file():
            raise FileNotFoundError(f"Digest template not found: {self.template_path}")

        tpl = DocxTemplate(str(self.template_path))
        context = self._build_context(
            digest,
            tpl,
            reviewer=reviewer,
            reviewer_approved=reviewer_approved,
            entry_date=entry_date,
        )
        # autoescape: LLM-generated content can contain XML-special chars —
        # a literal `<a>` in one chapter summary swallowed every later
        # section of the rendered document (DOC_066, chapter 45/106).
        tpl.render(context, autoescape=True)
        buf = io.BytesIO()
        tpl.save(buf)
        return buf.getvalue()
