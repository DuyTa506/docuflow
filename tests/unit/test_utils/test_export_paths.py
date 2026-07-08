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


def test_spatial_disabled_for_many_elements():
    use_spatial, embed = spatial_export_plan(
        _repo(elements=10_000, pages=50),
        "DOC_1",
        mode="auto",
        text_overridden=False,
    )
    assert use_spatial is False
    assert embed is False


def test_spatial_disabled_for_many_pages():
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
