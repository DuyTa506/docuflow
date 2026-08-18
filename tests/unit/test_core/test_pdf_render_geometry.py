"""Coordinate mapping for the hybrid PDF renderer."""

from core.pdf_render.geometry import Rect, stored_to_page_rect


class TestRect:
    def test_x_overlap_ratio(self):
        left = Rect(0, 0, 100, 50)
        right = Rect(120, 0, 220, 50)
        mid = Rect(50, 0, 150, 50)
        assert left.x_overlap_ratio(right) == 0.0
        assert left.x_overlap_ratio(mid) == 0.5

    def test_intersection_over_self(self):
        inner = Rect(10, 10, 30, 30)
        outer = Rect(0, 0, 100, 100)
        assert inner.intersection_over_self(outer) == 1.0
        assert outer.intersection_over_self(inner) < 0.1


class TestStoredToPageRect:
    def test_identity_at_72dpi(self):
        stored = Rect(10, 20, 110, 80)
        mapped = stored_to_page_rect(
            stored, image_w=595, image_h=842, page_w=595, page_h=842, rotation=0
        )
        assert mapped.x0 == 10
        assert mapped.y1 == 80

    def test_scales_ocr_raster_onto_page(self):
        stored = Rect(0, 0, 100, 200)
        mapped = stored_to_page_rect(
            stored, image_w=200, image_h=400, page_w=595, page_h=842, rotation=0
        )
        assert mapped.x1 == 297.5
        assert abs(mapped.y1 - 421.0) < 0.1

    def test_crop_rotation_180_flips(self):
        stored = Rect(10, 20, 30, 40)
        mapped = stored_to_page_rect(
            stored, image_w=100, image_h=100, page_w=100, page_h=100, rotation=180
        )
        assert mapped.x0 == 70
        assert mapped.y0 == 60
