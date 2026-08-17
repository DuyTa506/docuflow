from urllib.parse import quote

from utils.file_download import _content_disposition, build_docx_response


def test_content_disposition_uses_ascii_fallback_and_utf8_filename():
    filename = "Hướng dẫn rà soát băng thông.pdf"

    header = _content_disposition(filename)

    assert 'filename="Huong dan ra soat bang thong.pdf"' in header
    assert f"filename*=UTF-8''{quote(filename, safe='')}" in header
    # Starlette encodes every response header as Latin-1.
    header.encode("latin-1")


def test_docx_response_accepts_vietnamese_download_name():
    response = build_docx_response("Tổng thuật tài liệu.docx", "Nội dung")

    disposition = response.headers["content-disposition"]
    assert 'filename="Tong thuat tai lieu.docx"' in disposition
    assert "filename*=UTF-8''" in disposition
