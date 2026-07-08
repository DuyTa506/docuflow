"""Tests for digest template alignment."""
from core.spatial.reading_order import Edge, detect_cycles, build_reading_order_graph
from utils.digest_format import is_chapter_schema, join_catalog_items, bibliographic_defaults


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
