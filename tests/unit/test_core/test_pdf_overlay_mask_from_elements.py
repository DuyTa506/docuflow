"""Regression tests for build_layout_mask_from_elements -- the DB-backed
substitute for the YOLO layout model's translatable/reserved-region mask.

Confirmed live (real page comparison) that reusing already-extracted
Docling/OCR layout elements is at least as accurate as a fresh YOLO
detection pass, and removes a redundant model inference. This mask must
match the exact contract TranslateConverter expects: 0 = reserved
(non-translatable), 1 = untouched background, >=2 = a distinct
translatable text box -- with the same y-flip YOLO's own path already
applies, since stored bboxes are in the same top-down coordinate space
YOLO's raw detections use.
"""
from unittest.mock import MagicMock, patch

from core.pdf_overlay.pipeline import build_layout_mask_from_elements, _fetch_page_elements


class TestBuildLayoutMaskFromElements:
    def test_reserved_label_painted_zero(self):
        elements = [
            {"label": "table", "bbox_x1": 10, "bbox_y1": 10, "bbox_x2": 50, "bbox_y2": 30},
        ]
        box = build_layout_mask_from_elements(elements, page_h=100, page_w=100)
        # y-flip: top-down y10-30 -> bottom-up rows (100-30-1)=69 to (100-10+1)=91
        assert (box[69:91, 9:51] == 0).all()

    def test_translatable_label_gets_distinct_nonzero_index(self):
        elements = [
            {"label": "text", "bbox_x1": 10, "bbox_y1": 10, "bbox_x2": 50, "bbox_y2": 30},
            {"label": "title", "bbox_x1": 10, "bbox_y1": 40, "bbox_x2": 50, "bbox_y2": 60},
        ]
        box = build_layout_mask_from_elements(elements, page_h=100, page_w=100)
        v1 = box[80, 20]  # inside first element's flipped region
        v2 = box[50, 20]  # inside second element's flipped region
        assert v1 >= 2
        assert v2 >= 2
        assert v1 != v2

    def test_untouched_background_stays_one(self):
        elements = [
            {"label": "text", "bbox_x1": 10, "bbox_y1": 10, "bbox_x2": 20, "bbox_y2": 20},
        ]
        box = build_layout_mask_from_elements(elements, page_h=100, page_w=100)
        assert box[0, 0] == 1

    def test_reserved_wins_on_overlap(self):
        """Matches the original YOLO-path ordering: reserved labels are
        painted second, so they take precedence over any overlapping
        translatable box."""
        elements = [
            {"label": "text", "bbox_x1": 10, "bbox_y1": 10, "bbox_x2": 60, "bbox_y2": 60},
            {"label": "figure", "bbox_x1": 20, "bbox_y1": 20, "bbox_x2": 40, "bbox_y2": 40},
        ]
        box = build_layout_mask_from_elements(elements, page_h=100, page_w=100)
        # point inside the overlapping figure region (flipped)
        assert box[70, 30] == 0

    def test_unrecognized_labels_are_translatable_by_default(self):
        """Only the explicit reserved set (figure/table/equation/image) is
        excluded from translation -- title/text/sub_title/heading all get
        a normal translatable box."""
        elements = [
            {"label": "sub_title", "bbox_x1": 10, "bbox_y1": 10, "bbox_x2": 50, "bbox_y2": 30},
        ]
        box = build_layout_mask_from_elements(elements, page_h=100, page_w=100)
        assert box[80, 20] >= 2

    def test_empty_elements_returns_all_background(self):
        box = build_layout_mask_from_elements([], page_h=50, page_w=50)
        assert (box == 1).all()


class TestFetchPageElements:
    def test_groups_by_page_number(self):
        fake_page1 = MagicMock(page_number=1)
        fake_page2 = MagicMock(page_number=2)
        fake_el1 = MagicMock(label="text", bbox_x1=1, bbox_y1=2, bbox_x2=3, bbox_y2=4, page=fake_page1)
        fake_el2 = MagicMock(label="title", bbox_x1=5, bbox_y1=6, bbox_x2=7, bbox_y2=8, page=fake_page2)

        mock_repo = MagicMock()
        mock_repo.get_elements.return_value = [fake_el1, fake_el2]

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch("data.database.get_db_manager") as mock_dbm, \
             patch("data.repositories.DocumentRepository", return_value=mock_repo):
            mock_dbm.return_value.session.return_value = mock_session

            result = _fetch_page_elements("DOC_TEST")

        assert set(result.keys()) == {1, 2}
        assert result[1][0]["label"] == "text"
        assert result[2][0]["label"] == "title"

    def test_no_stored_elements_returns_empty_dict(self):
        mock_repo = MagicMock()
        mock_repo.get_elements.return_value = []

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch("data.database.get_db_manager") as mock_dbm, \
             patch("data.repositories.DocumentRepository", return_value=mock_repo):
            mock_dbm.return_value.session.return_value = mock_session

            result = _fetch_page_elements("DOC_TEST")

        assert result == {}
