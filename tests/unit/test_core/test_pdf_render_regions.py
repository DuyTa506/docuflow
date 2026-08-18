"""PageScene column / region classification."""

from core.pdf_render.geometry import PageMeta, Rect
from core.pdf_render.regions import build_page_scene, classify_role
from core.spatial.grouping import detect_columns_projection, group_into_lines, group_lines_to_blocks


def _elem(x1, y1, x2, y2, text, label="text", seq=0):
    return {
        "label": label,
        "text_content": text,
        "sequence_order": seq,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "page_number": 1,
    }


class TestClassifyRole:
    def test_narrow_vertical_is_passthrough(self):
        role = classify_role("text", "abc", Rect(10, 100, 16, 300), 595, 842)
        assert role == "vertical"

    def test_caption_from_text(self):
        role = classify_role("text", "Figure 3: pipeline", Rect(50, 400, 400, 420), 595, 842)
        assert role == "caption"


class TestTwoColumnScene:
    def test_does_not_merge_across_gutter(self):
        meta = PageMeta(page_number=1, width=595, height=842, image_width=595, image_height=842)
        elements = [
            _elem(40, 80, 250, 200, "Left column paragraph one.", seq=0),
            _elem(320, 80, 540, 200, "Right column paragraph one.", seq=1),
            _elem(40, 220, 250, 360, "Left column paragraph two.", seq=2),
            _elem(320, 220, 540, 360, "Right column paragraph two.", seq=3),
        ]
        scene = build_page_scene(elements, meta, source_hint="docling")
        assert len(scene.columns) >= 2
        left = [r for r in scene.regions if r.bbox.cx < 300]
        right = [r for r in scene.regions if r.bbox.cx > 300]
        assert left and right
        assert {r.column_index for r in left} != {r.column_index for r in right}

    def test_full_width_title_excluded_from_column_histogram(self):
        elems = [
            {
                "bbox_x1": 40,
                "bbox_y1": 40,
                "bbox_x2": 555,
                "bbox_y2": 70,
                "label": "title",
            },
            {
                "bbox_x1": 40,
                "bbox_y1": 100,
                "bbox_x2": 250,
                "bbox_y2": 400,
                "label": "text",
            },
            {
                "bbox_x1": 320,
                "bbox_y1": 100,
                "bbox_x2": 540,
                "bbox_y2": 400,
                "label": "text",
            },
        ]
        cols = detect_columns_projection(elems, 595)
        assert len(cols) == 2


class TestCenterGapFallback:
    def test_tight_gutter_two_column_page(self):
        elems = [
            {"bbox_x1": 37, "bbox_y1": 60, "bbox_x2": 292, "bbox_y2": 160, "label": "text"},
            {"bbox_x1": 303, "bbox_y1": 60, "bbox_x2": 558, "bbox_y2": 220, "label": "text"},
            {"bbox_x1": 37, "bbox_y1": 170, "bbox_x2": 292, "bbox_y2": 270, "label": "text"},
            {"bbox_x1": 303, "bbox_y1": 230, "bbox_x2": 558, "bbox_y2": 400, "label": "text"},
        ]
        cols = detect_columns_projection(elems, 595)
        assert len(cols) == 2
        assert cols[0].x2 < 320
        assert cols[1].x1 > 280


class TestDuplicateRegionDrop:
    def test_later_exact_copy_is_removed(self):
        meta = PageMeta(page_number=1, width=595, height=842, image_width=595, image_height=842)
        text = "However, producing SQL queries requires computational expertise and time to learn a new database. " * 3
        elements = [
            _elem(303, 60, 558, 230, text, seq=0),
            _elem(37, 640, 292, 760, text, seq=1),
        ]
        scene = build_page_scene(elements, meta, source_hint="docling")
        hits = [r for r in scene.regions if "producing SQL" in r.text]
        assert len(hits) == 1
        assert hits[0].bbox.y0 < 100
        blanks = [r for r in scene.regions if not r.text and r.bbox.y0 > 500]
        assert blanks


class TestGroupLinesNeverCrossesColumns:
    def test_interleaved_lines_stay_split(self):
        lines = group_into_lines(
            [
                {"bbox_x1": 40, "bbox_y1": 80, "bbox_x2": 250, "bbox_y2": 100, "label": "text"},
                {"bbox_x1": 320, "bbox_y1": 80, "bbox_x2": 540, "bbox_y2": 100, "label": "text"},
                {"bbox_x1": 40, "bbox_y1": 110, "bbox_x2": 250, "bbox_y2": 130, "label": "text"},
                {"bbox_x1": 320, "bbox_y1": 110, "bbox_x2": 540, "bbox_y2": 130, "label": "text"},
            ]
        )
        # Two columns on the same baseline may share a line group; block
        # grouping must still split by x-overlap.
        blocks = group_lines_to_blocks(lines)
        xs = [b.bbox["x1"] for b in blocks]
        assert min(xs) < 100
        assert max(xs) > 250
