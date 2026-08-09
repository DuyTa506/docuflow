"""Tests for digest template alignment."""

from core.spatial.reading_order import Edge, build_reading_order_graph, detect_cycles
from utils.digest_format import bibliographic_defaults, is_chapter_schema, join_catalog_items


class TestDetectCyclesIterative:
    def test_no_recursion_error_on_chain(self):
        """Long chain must not hit Python recursion limit."""
        n = 2000
        graph = {f"n{i}": [] for i in range(n)}
        for i in range(n - 1):
            graph[f"n{i}"].append(Edge(f"n{i}", f"n{i+1}", relation="before"))

        cycles = detect_cycles(graph)
        assert cycles == []

    def test_detects_simple_cycle(self):
        graph = {
            "a": [Edge("a", "b", relation="before")],
            "b": [Edge("b", "c", relation="before")],
            "c": [Edge("c", "a", relation="before")],
        }
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1


class TestDigestFormat:
    def test_is_chapter_schema(self):
        assert is_chapter_schema({"chapters": [{"number": 1}]})
        assert not is_chapter_schema({"key_points": []})
        assert not is_chapter_schema(None)

    def test_join_catalog_items(self):
        assert join_catalog_items(["A", "B"]) == "A; B"
        assert join_catalog_items([]) == ""

    def test_bibliographic_defaults(self):
        d = bibliographic_defaults(title="T", pages=100)
        assert d["title_display"] == "T"
        assert d["pages"] == "100"


class TestDigestRenderer:
    def test_render_produces_docx_bytes(self):
        import importlib

        digest_svc = importlib.import_module("services.digest_service")
        digest_rnd = importlib.import_module("services.digest_renderer")

        DigestResult = digest_svc.DigestResult
        ChapterEntry = digest_svc.ChapterEntry
        KeywordEntry = digest_svc.KeywordEntry
        DigestRenderer = digest_rnd.DigestRenderer

        digest = DigestResult(
            document_id="DOC_001",
            title="Test Book",
            source_language="en",
            original_filename="test.pdf",
            bibliographic=bibliographic_defaults(title="Test Book (Sách thử)", pages=10),
            abstract="Tóm tắt thử nghiệm.",
            chapters=[
                ChapterEntry(
                    number=1,
                    title_vi="Giới thiệu",
                    title_original="Introduction",
                    content="Nội dung chương 1.",
                )
            ],
            keywords=[KeywordEntry(keyword="radar", display="Radar (radar)", weight=0.9)],
            usage_scope={
                "undergraduate": ["Ngành Kỹ thuật điện tử"],
                "master": [],
                "phd": [],
                "strong_research_groups": [],
            },
            research_directions=[],
        )
        renderer = DigestRenderer()
        data = renderer.render(digest)
        assert data[:2] == b"PK"
        assert len(data) > 1000


class TestDigestRendererXmlEscaping:
    def test_chapter_content_with_html_tags_does_not_swallow_later_chapters(self):
        """Regression (DOC_066): chapter 45's summary literally contained the
        HTML tag `<a>` (the chapter is about counting link tags with regex).
        docxtpl renders context values into raw document XML with
        autoescape=False by default, so the `<a>` opened a bogus XML element
        that swallowed every later chapter plus §2.3/§3 — the exported digest
        (DOCX and the PDF built from it) silently ended at chapter 45 of 106,
        mid-sentence."""
        import importlib
        import io

        from docx import Document

        digest_svc = importlib.import_module("services.digest_service")
        digest_rnd = importlib.import_module("services.digest_renderer")

        digest = digest_svc.DigestResult(
            document_id="DOC_001",
            title="Test Book",
            source_language="en",
            original_filename="test.pdf",
            bibliographic={"title_display": "Test & Book <2013>", "pages": "10"},
            abstract="Tóm tắt.",
            chapters=[
                digest_svc.ChapterEntry(
                    number=1,
                    title_vi="Đếm thẻ liên kết",
                    title_original="Counting link tags",
                    content="Dùng regex để đếm các thẻ <a> trong văn bản thô.",
                ),
                digest_svc.ChapterEntry(
                    number=2,
                    title_vi="Chương sau",
                    title_original="Later chapter",
                    content="Nội dung sau thẻ HTML ở chương trước phải sống sót.",
                ),
            ],
            keywords=[digest_svc.KeywordEntry(keyword="html", display="HTML <tags>", weight=0.5)],
            usage_scope={
                "undergraduate": [],
                "master": [],
                "phd": [],
                "strong_research_groups": [],
            },
            research_directions=[],
        )

        data = digest_rnd.DigestRenderer().render(digest)
        text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)

        assert "các thẻ <a> trong văn bản thô" in text
        assert "phải sống sót" in text
        assert "Test & Book <2013>" in text
