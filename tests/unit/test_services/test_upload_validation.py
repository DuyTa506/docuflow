"""Upload-time PDF validation.

Regression (DOC_068): a password-protected PDF was accepted silently with
total_pages=0; extraction then failed on every Temporal retry with an opaque
per-page "document closed or encrypted", cascading into a FAILED digest.
Broken input must be rejected at upload with an actionable message.
"""

import pymupdf
import pytest

from services.document_service import _validate_pdf_readable


def _make_pdf(path, *, pages=2, user_pw=None):
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Trang {i + 1}")
    if user_pw:
        doc.save(
            path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw=user_pw,
        )
    else:
        doc.save(path)
    doc.close()


class TestValidatePdfReadable:
    def test_normal_pdf_returns_page_count(self, tmp_path):
        p = tmp_path / "ok.pdf"
        _make_pdf(str(p), pages=3)
        assert _validate_pdf_readable(str(p)) == 3

    def test_password_protected_pdf_rejected_with_actionable_message(self, tmp_path):
        p = tmp_path / "locked.pdf"
        _make_pdf(str(p), pages=2, user_pw="secret")
        with pytest.raises(ValueError) as exc:
            _validate_pdf_readable(str(p))
        assert "mật khẩu" in str(exc.value)

    def test_garbage_bytes_rejected(self, tmp_path):
        p = tmp_path / "broken.pdf"
        p.write_bytes(b"%PDF-1.6 this is not really a pdf at all")
        with pytest.raises(ValueError):
            _validate_pdf_readable(str(p))
