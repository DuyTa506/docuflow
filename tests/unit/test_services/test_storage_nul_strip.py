"""Postgres rejects NUL (0x00) in text columns; storage must strip it.

Live regression (DOC_003, Ballistics): a PDF text layer carried ``\\x00`` and
``save_unified_elements`` raised ``ValueError: A string literal cannot contain
NUL (0x00) characters`` on every Temporal retry, failing the whole book.
"""

from unittest.mock import MagicMock, patch

from data.db_models import LayoutElement, Page
from services.storage_service import DocumentStorageService
from serving.logic import ServicePageResult


def _storage_with_no_existing_page():
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = None
    storage = DocumentStorageService(session)
    return storage, session


def _added(session, cls):
    return [c.args[0] for c in session.add.call_args_list if isinstance(c.args[0], cls)]


def test_save_unified_elements_strips_nul_from_page_and_elements():
    storage, session = _storage_with_no_existing_page()
    with patch.object(storage, "_upload_page_image", return_value=(None, None)):
        storage.save_unified_elements(
            document_id="DOC_1",
            page_number=1,
            markdown_content="Bal\x00listics",
            layout_dicts=[{"label": "text", "text_content": "ex\x00terior", "bbox_x1": 0}],
            page_type="text",
        )

    (page,) = _added(session, Page)
    (elem,) = _added(session, LayoutElement)
    assert page.markdown_content == "Ballistics"
    assert elem.text_content == "exterior"


def test_save_page_result_strips_nul():
    storage, session = _storage_with_no_existing_page()
    result = ServicePageResult(
        page_num=1,
        markdown="a\x00b",
        layout_elements=[{"label": "text", "text_content": "c\x00d"}],
    )
    storage.save_page_result("DOC_1", result)

    (page,) = _added(session, Page)
    (elem,) = _added(session, LayoutElement)
    assert page.markdown_content == "ab"
    assert elem.text_content == "cd"


def test_existing_page_update_strips_nul():
    existing = Page(id="PG_1", document_id="DOC_1", page_number=1, markdown_content="old")
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = existing
    storage = DocumentStorageService(session)
    with patch.object(storage, "_replace_layout_elements"):
        storage.save_unified_elements("DOC_1", 1, "x\x00y", [], page_type="text")
    assert existing.markdown_content == "xy"
