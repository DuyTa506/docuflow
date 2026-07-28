"""Export path selection for large documents."""

from unittest.mock import MagicMock, patch

from utils.export_paths import (
    resolve_ocr_content,
    spatial_export_plan,
    translation_spatial_plan,
)


def _repo(*, elements: int = 100, pages: int = 50):
    repo = MagicMock()
    repo.count_elements.return_value = elements
    repo.count_pages.return_value = pages
    repo.get_pages.return_value = []
    return repo


def test_resolve_ocr_prefers_offloaded_blob():
    repo = MagicMock()
    dt = MagicMock(ocr_content=None, ocr_content_key="documents/DOC_1/ocr.md")
    with patch("utils.export_paths.read_text_field", return_value="full ocr blob"):
        assert resolve_ocr_content(repo, "DOC_1", dt, content_type="ocr") == "full ocr blob"
    repo.get_pages.assert_not_called()


def test_spatial_disabled_for_many_elements(monkeypatch):
    # caps are deployment-tunable via .env — pin them so the test is hermetic
    monkeypatch.setattr("utils.export_paths.settings.ocr_download_spatial_max_elements", 2500)
    use_spatial, embed = spatial_export_plan(
        _repo(elements=10_000, pages=50),
        "DOC_1",
        mode="auto",
        text_overridden=False,
    )
    assert use_spatial is False
    assert embed is False


def test_spatial_disabled_for_many_pages(monkeypatch):
    monkeypatch.setattr("utils.export_paths.settings.ocr_download_spatial_max_pages", 200)
    use_spatial, embed = spatial_export_plan(
        _repo(elements=500, pages=500),
        "DOC_1",
        mode="auto",
        text_overridden=False,
    )
    assert use_spatial is False


def test_translation_spatial_disabled_for_huge_element_lists():
    use_spatial, embed = translation_spatial_plan(
        _repo(elements=50_000, pages=50),
        "DOC_1",
        element_count=50_000,
        source="auto",
    )
    assert use_spatial is False
    assert embed is False


def test_spatial_text_only_when_many_elements_but_under_cap():
    with patch("utils.export_paths.settings") as mock_settings:
        mock_settings.ocr_download_spatial_max_elements = 2500
        mock_settings.ocr_download_spatial_max_pages = 200
        mock_settings.export_spatial_embed_images_max_elements = 800
        use_spatial, embed = spatial_export_plan(
            _repo(elements=1500, pages=100),
            "DOC_1",
            mode="auto",
            text_overridden=False,
        )
    assert use_spatial is True
    assert embed is False


class TestTranslationRoutingSpatialCap:
    def test_translation_routing_not_capped_by_download_limit(self):
        """Regression (DOC_066, 290-page book, 2534 elements): translation
        routing reused ocr_download_spatial_max_elements (2500 — a
        download-speed cap) as its eligibility gate, so any book crossing it
        silently fell to tree/flat mode and the translated artifact lost
        every figure ("(img_content)[figure]" leaked as literal text) even
        though all 145 crop images sat in MinIO. BlockTranslator exists
        precisely for large element lists — translation gets its own, far
        higher runaway guard."""
        from utils.export_paths import translation_routing_allows_spatial

        assert translation_routing_allows_spatial(2534) is True
        assert translation_routing_allows_spatial(0) is False
        # the dedicated guard still exists for true runaway documents
        assert translation_routing_allows_spatial(100_000) is False

    def test_translation_service_uses_dedicated_cap(self):
        """The download cap must no longer appear in translation routing."""
        import inspect

        import services.translation_service as ts

        src = inspect.getsource(ts)
        assert "ocr_download_spatial_max_elements" not in src
        assert "translation_routing_allows_spatial" in src
