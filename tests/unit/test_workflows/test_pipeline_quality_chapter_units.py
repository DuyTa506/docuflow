"""A fragmented §2.2 shipped with a clean bill of health.

`chapter_count` was recorded in the quality report but never checked, and
`research_directions` was not checked at all — so the N4.11.160 run reported
success with 265 §2.2 entries, an empty §3, and no warning of any kind.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from config.settings import settings
from services.pipeline.quality import build_quality_report


def _keyword(i, bilingual=True):
    name = f"term{i}"
    return SimpleNamespace(
        keyword=name, display=f"Thuật ngữ {i} ({name})" if bilingual else name, weight=0.9
    )


def _digest(chapter_count=12, missing=None, keywords=None, source_language="ru"):
    return SimpleNamespace(
        chapters=[{"number": i} for i in range(chapter_count)],
        missing=list(missing or []),
        keywords=[_keyword(i) for i in range(20)] if keywords is None else keywords,
        source_language=source_language,
    )


def _run(details, *, chapter_count=12, missing=None, keywords=None, source_language="ru"):
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    # One stand-in serves both the MainContent and the Summary query; the 12
    # sentences keep the unrelated abstract-length gate quiet.
    main_content = SimpleNamespace(
        details=details,
        status="COMPLETED",
        content=" ".join(f"Câu {i}." for i in range(1, 13)),
    )
    query = session.query.return_value.filter.return_value
    query.first.return_value = object()  # TreeIndex present
    query.order_by.return_value.first.return_value = main_content

    with (
        patch("services.pipeline.quality.get_db_manager") as db_manager,
        patch("services.pipeline.quality.DigestService") as digest_service,
        patch("services.pipeline.quality.is_chapter_schema", return_value=True),
    ):
        db_manager.return_value.session.return_value = session
        digest_service.return_value.assemble.return_value = _digest(
            chapter_count, missing, keywords, source_language
        )
        return build_quality_report("DOC_001")


def _joined(report):
    return " | ".join(report["warnings"])


class TestFragmentationWarnings:
    def test_warns_when_unit_count_exceeds_cap(self):
        report = _run({"unit_selection_tier": "root_children", "max_units": 18}, chapter_count=265)

        assert "phân mạch" in _joined(report)
        assert report["chapter_count"] == 265

    def test_uses_the_adaptive_cap_recorded_by_the_run(self):
        """The ceiling scales with document length — settings alone can't say."""
        details = {"unit_selection_tier": "chapter_vocabulary", "max_units": 12}

        assert "phân mạch" in _joined(_run(details, chapter_count=13))
        assert "phân mạch" not in _joined(_run(details, chapter_count=11))

    def test_warns_when_units_are_too_small(self):
        report = _run(
            {"unit_selection_tier": "chapter_vocabulary", "median_unit_chars": 300},
        )

        assert "phân mảnh" in _joined(report)

    def test_warns_on_machine_cut_units(self):
        report = _run({"unit_selection_tier": "mass_segmentation", "median_unit_chars": 20000})

        assert "khối lượng" in _joined(report)

    def test_warns_on_low_coverage(self):
        report = _run(
            {
                "unit_selection_tier": "chapter_vocabulary",
                "median_unit_chars": 20000,
                "coverage_ratio": 0.35,
            }
        )

        assert "bao phủ" in _joined(report)

    def test_healthy_digest_raises_no_unit_warnings(self):
        report = _run(
            {
                "unit_selection_tier": "chapter_vocabulary",
                "median_unit_chars": 40000,
                "coverage_ratio": 0.98,
            }
        )

        joined = _joined(report)
        for phrase in ("phân mạch", "phân mảnh", "khối lượng", "bao phủ"):
            assert phrase not in joined, joined
        assert report["unit_selection_tier"] == "chapter_vocabulary"

    def test_selection_metadata_is_reported(self):
        report = _run(
            {
                "unit_selection_tier": "numbered_sections",
                "median_unit_chars": 5000,
                "coverage_ratio": 0.9,
            }
        )

        assert report["unit_selection_tier"] == "numbered_sections"
        assert report["median_unit_chars"] == 5000
        assert report["coverage_ratio"] == 0.9


def test_missing_research_directions_now_warns():
    report = _run({"unit_selection_tier": "chapter_vocabulary"}, missing=["research_directions"])

    assert "hướng nghiên cứu" in _joined(report).lower()


class TestUsageScopeWarning:
    """An empty §3 means two different things; only one is actionable."""

    def test_says_so_when_no_catalog_was_ever_loaded(self):
        with patch("services.pipeline.quality.load_catalog", return_value={}):
            report = _run({"unit_selection_tier": "chapter_vocabulary"}, missing=["usage_scope"])

        assert "chưa nạp danh mục" in _joined(report)

    def test_plain_warning_when_the_catalog_exists_but_nothing_matched(self):
        with patch(
            "services.pipeline.quality.load_catalog",
            return_value={
                "undergraduate": [
                    {
                        "code": "74801",
                        "name": "Máy tính",
                        "children": [{"code": "7480101", "name": "Khoa học máy tính"}],
                    }
                ]
            },
        ):
            report = _run({"unit_selection_tier": "chapter_vocabulary"}, missing=["usage_scope"])

        joined = _joined(report)
        assert "§3 phạm vi CTĐT/NNC trống" in joined
        assert "chưa nạp danh mục" not in joined


class TestKeywordQuantityAndForm:
    """Mẫu yêu cầu 20 từ khoá dạng `Tiếng Việt (nguyên bản)`; `LIMIT 20` chỉ là trần."""

    def test_warns_when_fewer_than_twenty_keywords(self):
        report = _run(
            {"unit_selection_tier": "chapter_vocabulary"},
            keywords=[_keyword(i) for i in range(7)],
        )

        assert "7/20 từ khóa" in _joined(report)

    def test_no_warning_at_twenty(self):
        assert "/20 từ khóa" not in _joined(_run({"unit_selection_tier": "chapter_vocabulary"}))

    def test_warns_when_the_bilingual_form_is_missing(self):
        report = _run(
            {"unit_selection_tier": "chapter_vocabulary"},
            keywords=[_keyword(i, bilingual=False) for i in range(20)],
        )

        assert "song ngữ" in _joined(report)

    def test_vietnamese_source_needs_no_bilingual_form(self):
        report = _run(
            {"unit_selection_tier": "chapter_vocabulary"},
            keywords=[_keyword(i, bilingual=False) for i in range(20)],
            source_language="vi",
        )

        assert "song ngữ" not in _joined(report)


@pytest.mark.parametrize("cap", [10, 50])
def test_settings_override_applies_when_the_run_recorded_no_cap(monkeypatch, cap):
    """Older MainContent rows predate `max_units` in details."""
    monkeypatch.setattr(settings, "main_content_max_units", cap)

    report = _run({"unit_selection_tier": "root_children"}, chapter_count=cap + 1)

    assert "phân mạch" in _joined(report)
