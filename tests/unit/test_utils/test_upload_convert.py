"""Uploads that the pipeline cannot read directly, converted at upload time.

The E2E batch (2026-09-20) rejected four Russian DjVu scans — the format
Russian technical libraries publish in — with "Unsupported file type". TIFF
scans, ODT/RTF papers, slide decks and EPUB books had the same problem. Each
converts to something extraction already handles, so convert instead of
refusing.
"""

import os

import pytest
from PIL import Image

from utils import upload_convert
from utils.upload_convert import (
    UPLOAD_EXTENSIONS,
    convert_upload,
    converted_extension,
)


class TestWhichFormatsConvert:
    @pytest.mark.parametrize("ext", [".pdf", ".png", ".jpg", ".jpeg", ".docx", ".doc"])
    def test_formats_the_pipeline_reads_are_left_alone(self, ext):
        assert converted_extension(ext) is None

    @pytest.mark.parametrize(
        "ext,target",
        [
            (".djvu", ".pdf"),
            (".djv", ".pdf"),
            (".tif", ".png"),
            (".tiff", ".png"),
            (".bmp", ".png"),
            (".webp", ".png"),
            (".odt", ".docx"),
            (".rtf", ".docx"),
            (".pptx", ".pdf"),
            (".ppt", ".pdf"),
            (".epub", ".docx"),
            (".txt", ".docx"),
            (".md", ".docx"),
        ],
    )
    def test_convertible_formats_map_to_a_readable_one(self, ext, target):
        assert converted_extension(ext) == target

    def test_an_unknown_format_is_not_convertible(self):
        assert converted_extension(".exe") is None

    def test_upload_extensions_cover_both_groups(self):
        assert {".pdf", ".docx"} <= UPLOAD_EXTENSIONS
        assert {".djvu", ".tiff", ".epub"} <= UPLOAD_EXTENSIONS
        assert ".exe" not in UPLOAD_EXTENSIONS

    def test_extensions_are_matched_case_insensitively(self):
        assert converted_extension(".DJVU") == ".pdf"


class TestConverting:
    def test_a_pdf_is_returned_unchanged(self, tmp_path):
        src = tmp_path / "book.pdf"
        src.write_bytes(b"%PDF-1.4\n")
        assert convert_upload(str(src)) == str(src)
        assert src.exists()

    def test_a_tiff_scan_becomes_a_png(self, tmp_path):
        src = tmp_path / "scan.tiff"
        Image.new("RGB", (8, 8), "white").save(src)

        out = convert_upload(str(src))

        assert out.endswith(".png")
        assert Image.open(out).size == (8, 8)
        # The source is a temp upload; leaving it behind fills UPLOAD_DIR.
        assert not src.exists()

    def test_a_djvu_scan_is_decoded_page_by_page_into_a_pdf(self, tmp_path, monkeypatch):
        import fitz

        src = tmp_path / "scan.djvu"
        src.write_bytes(b"AT&TFORM")
        calls = []
        live = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            Image.new("RGB", (6, 9), "white").save(cmd[-1])
            # Decoded pages must not pile up: a 500-page book at 200 DPI is
            # gigabytes of TIFF if every page is written before the merge.
            live.append(len([f for f in os.listdir(os.path.dirname(cmd[-1]))]))

        monkeypatch.setattr(upload_convert, "_run", fake_run)
        monkeypatch.setattr(upload_convert, "_djvu_page_count", lambda p: 3)

        out = convert_upload(str(src))

        assert out == str(tmp_path / "scan.pdf")
        assert [c[0] for c in calls] == ["ddjvu"] * 3
        assert [c for c in calls[0] if c.startswith("-page=")] == ["-page=1"]
        assert [c for c in calls[2] if c.startswith("-page=")] == ["-page=3"]
        assert max(live) <= 2  # the page being decoded, plus its re-encode
        with fitz.open(out) as pdf:
            assert pdf.page_count == 3
        assert not src.exists()

    def test_a_page_that_cannot_be_decoded_does_not_lose_the_book(self, tmp_path, monkeypatch):
        import fitz

        src = tmp_path / "scan.djvu"
        src.write_bytes(b"AT&TFORM")

        def fake_run(cmd, **kwargs):
            if "-page=2" in cmd:
                raise RuntimeError("ddjvu: corrupt page")
            Image.new("RGB", (6, 9), "white").save(cmd[-1])

        monkeypatch.setattr(upload_convert, "_run", fake_run)
        monkeypatch.setattr(upload_convert, "_djvu_page_count", lambda p: 3)

        out = convert_upload(str(src))

        with fitz.open(out) as pdf:
            assert pdf.page_count == 2

    def test_an_odt_paper_goes_through_libreoffice(self, tmp_path, monkeypatch):
        src = tmp_path / "paper.odt"
        src.write_bytes(b"PK\x03\x04")
        seen = {}

        def fake_soffice(args, **kwargs):
            seen["args"] = args
            out_dir = args[args.index("--outdir") + 1]
            open(os.path.join(out_dir, "paper.docx"), "wb").write(b"PK\x03\x04")
            return type("R", (), {"returncode": 0, "stderr": ""})()

        monkeypatch.setattr(upload_convert, "run_soffice", fake_soffice)

        out = convert_upload(str(src))

        assert out == str(tmp_path / "paper.docx")
        assert seen["args"][:3] == ["--headless", "--convert-to", "docx"]

    def test_an_epub_goes_through_pandoc(self, tmp_path, monkeypatch):
        src = tmp_path / "notes.epub"
        src.write_bytes(b"PK\x03\x04")
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            open(cmd[-1], "wb").write(b"PK\x03\x04")

        monkeypatch.setattr(upload_convert, "_run", fake_run)

        out = convert_upload(str(src))

        assert out.endswith("notes.docx")
        assert seen["cmd"][0] == "pandoc"


class TestWhenTheToolIsMissing:
    def test_the_error_names_the_package_to_install(self, tmp_path, monkeypatch):
        src = tmp_path / "scan.djvu"
        src.write_bytes(b"AT&TFORM")
        monkeypatch.setattr(upload_convert.shutil, "which", lambda name: None)

        with pytest.raises(RuntimeError) as err:
            convert_upload(str(src))

        assert "djvulibre-bin" in str(err.value)
        # The upload is kept: a retry after installing the tool should work.
        assert src.exists()

    def test_a_failed_conversion_reports_the_source_format(self, tmp_path, monkeypatch):
        src = tmp_path / "scan.djvu"
        src.write_bytes(b"not a djvu file")
        monkeypatch.setattr(upload_convert, "_run", lambda cmd, **kw: None)

        with pytest.raises(RuntimeError) as err:
            convert_upload(str(src))

        assert ".djvu" in str(err.value)
