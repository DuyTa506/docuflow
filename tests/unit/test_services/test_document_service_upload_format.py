"""The stored file decides the format, not the name the user uploaded.

A DjVu upload is converted to PDF before it is registered (see
``utils.upload_convert``). Keying the format off ``original_filename`` made
those rows format="unknown" with total_pages=0, which extraction cannot
route.
"""

from unittest.mock import MagicMock, patch

import pytest

from services.document_service import DocumentService


@pytest.fixture
def db():
    session = MagicMock()
    session.add = MagicMock()
    session.commit = MagicMock()
    session.refresh = MagicMock()
    return session


def _upload(db, tmp_path, stored_name, original_filename, pages=7):
    path = tmp_path / stored_name
    path.write_bytes(b"%PDF-1.4\n")
    storage = MagicMock()
    with (
        patch("services.document_service.IdGenerator") as ids,
        patch("services.object_storage.get_object_storage", return_value=storage),
        patch("services.document_service._validate_pdf_readable", return_value=pages),
    ):
        ids.next_id.return_value = "DOC_099"
        doc = DocumentService().upload_document(
            db,
            file_path_on_disk=str(path),
            original_filename=original_filename,
            user_id="USR_001",
        )
    return doc, storage


class TestFormatFollowsTheStoredFile:
    def test_converted_djvu_is_registered_as_a_pdf(self, db, tmp_path):
        doc, storage = _upload(db, tmp_path, "_upload_x.pdf", "Определение-координат.djvu")

        assert (doc.format, doc.file_type) == ("pdf", "pdf")
        assert doc.total_pages == 7
        # The user still recognises their file in the list.
        assert doc.original_filename == "Определение-координат.djvu"
        # The object key must carry the real extension, not .djvu.
        assert storage.put_file.call_args.args[0].endswith(".pdf")

    def test_a_plain_pdf_upload_is_unchanged(self, db, tmp_path):
        doc, _ = _upload(db, tmp_path, "_upload_y.pdf", "book.pdf")

        assert (doc.format, doc.file_type, doc.total_pages) == ("pdf", "pdf", 7)

    def test_a_converted_slide_deck_keeps_its_original_name(self, db, tmp_path):
        doc, storage = _upload(db, tmp_path, "_upload_z.pdf", "lecture 3.pptx")

        assert doc.format == "pdf"
        assert doc.original_filename == "lecture 3.pptx"
        assert storage.put_file.call_args.args[0].endswith("lecture_3.pdf")
