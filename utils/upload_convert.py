"""Turn an uploaded file into something the extraction pipeline can read.

Extraction handles PDF, images, DOC and DOCX. Everything else used to be
refused at the door, which cost the 2026-09 E2E four Russian DjVu scans —
the format Russian technical libraries publish in. Each rejected format has
a cheap, lossless-enough conversion to one of the four, so do that instead:

    .djvu/.djv              → .pdf   (ddjvu, djvulibre)
    .tif/.tiff/.bmp/.webp   → .png   (Pillow)
    .odt/.rtf               → .docx  (LibreOffice)
    .ppt/.pptx              → .pdf   (LibreOffice)
    .epub/.txt/.md          → .docx  (pandoc)

The converted file keeps the upload's name and directory; only the suffix
changes. The document row still shows the name the user uploaded.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

from utils.soffice import run_soffice

logger = logging.getLogger(__name__)

# What extraction reads without help (services/extractors, document_service).
PIPELINE_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".docx", ".doc"})

# source extension → extension we convert it to
CONVERSIONS: dict[str, str] = {
    ".djvu": ".pdf",
    ".djv": ".pdf",
    ".tif": ".png",
    ".tiff": ".png",
    ".bmp": ".png",
    ".webp": ".png",
    ".odt": ".docx",
    ".rtf": ".docx",
    ".ppt": ".pdf",
    ".pptx": ".pdf",
    ".epub": ".docx",
    ".txt": ".docx",
    ".md": ".docx",
}

# Everything the upload endpoint accepts.
UPLOAD_EXTENSIONS = frozenset(PIPELINE_EXTENSIONS | set(CONVERSIONS))

# Tool → the package that provides it, so the error tells the admin what to do.
_INSTALL_HINT = {
    "ddjvu": "djvulibre-bin",
    "pandoc": "pandoc",
    "soffice": "libreoffice",
}

_TIMEOUT_SECONDS = 600

# DjVu render resolution: enough for OCR, a tenth of the native size.
_SCAN_DPI = 200


def converted_extension(ext: str) -> Optional[str]:
    """The extension ``ext`` converts to, or None when no conversion is needed."""
    return CONVERSIONS.get((ext or "").lower())


def _run(cmd: list[str], **kwargs) -> None:
    """Run a converter, raising RuntimeError on a non-zero exit."""
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, **kwargs)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip()[:500])


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        hint = _INSTALL_HINT.get(tool, tool)
        raise RuntimeError(
            f"'{tool}' is not installed (apt install {hint}); " "or upload the file as PDF instead."
        )


def _to_png(src: str, dst: str) -> None:
    from PIL import Image

    with Image.open(src) as img:
        # PNG has no CMYK/16-bit-palette equivalent the OCR path reads back.
        img.convert("RGB").save(dst, format="PNG")


def _compress_scan(tif_path: str) -> str:
    """Re-encode one decoded page. ddjvu writes full-resolution RGB TIFF —
    8 MB a page for a book whose whole DjVu is 1 MB. Bitonal text stays
    1-bit PNG (sharp, tiny); anything else becomes JPEG."""
    from PIL import Image

    with Image.open(tif_path) as img:
        if img.mode == "1":
            out = tif_path + ".png"
            img.save(out, format="PNG", optimize=True)
        else:
            out = tif_path + ".jpg"
            img.convert("L" if img.mode in ("L", "LA") else "RGB").save(
                out, format="JPEG", quality=80, optimize=True
            )
    return out


def _djvu_page_count(src: str) -> int:
    """Pages in a DjVu document, via djvused (ships with ddjvu)."""
    result = subprocess.run(
        ["djvused", "-e", "n", src],
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        return int((result.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"djvused could not read the page count: {exc}") from exc


def _to_pdf_via_ddjvu(src: str, dst: str) -> None:
    """DjVu → PDF. ddjvu itself writes images only (pnm/tiff), so decode to a
    TIFF per page and staple those into a PDF, one page at a time — a 500-page
    scan held in memory at once is gigabytes."""
    import fitz

    _require("ddjvu")
    total = _djvu_page_count(src)
    with tempfile.TemporaryDirectory(prefix="djvu_") as tmp:
        pdf = fitz.open()
        decoded = 0
        try:
            for number in range(1, total + 1):
                raw = os.path.join(tmp, "page.tif")
                # One page at a time: `-eachpage` writes the whole book first,
                # which is ~3 MB × pages of TIFF sitting in /tmp (1.5 GB for a
                # 500-page scan). -scale: native is ~430 DPI, which OCR
                # downsamples anyway; 200 DPI is sharp at a tenth of the size.
                try:
                    _run(
                        [
                            "ddjvu",
                            "-format=tiff",
                            f"-scale={_SCAN_DPI}",
                            f"-page={number}",
                            src,
                            raw,
                        ]
                    )
                except RuntimeError as exc:
                    # One unreadable page must not cost the whole book.
                    logger.warning("ddjvu skipped page %d/%d: %s", number, total, exc)
                    continue
                page_file = _compress_scan(raw)
                os.remove(raw)
                decoded += 1
                with fitz.open(page_file) as img:
                    # Size the page from the render DPI, or every page comes
                    # out metres wide (PyMuPDF assumes 96 DPI for bare images).
                    rect = fitz.Rect(
                        0,
                        0,
                        img[0].rect.width * 72 / _SCAN_DPI,
                        img[0].rect.height * 72 / _SCAN_DPI,
                    )
                    page = pdf.new_page(width=rect.width, height=rect.height)
                    page.insert_image(rect, filename=page_file)
                os.remove(page_file)
            if not decoded:
                raise RuntimeError("ddjvu decoded no pages")
            pdf.save(dst, deflate=True, garbage=3)
        finally:
            pdf.close()


def _via_soffice(src: str, dst: str) -> None:
    _require("soffice")
    target = os.path.splitext(dst)[1].lstrip(".")
    with tempfile.TemporaryDirectory(prefix="upload_convert_") as out_dir:
        result = run_soffice(
            ["--headless", "--convert-to", target, "--outdir", out_dir, src],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        if getattr(result, "returncode", 1) != 0:
            raise RuntimeError((getattr(result, "stderr", "") or "").strip()[:500])
        produced = os.path.join(out_dir, os.path.splitext(os.path.basename(src))[0] + f".{target}")
        if not os.path.exists(produced):
            raise RuntimeError(f"LibreOffice produced no .{target} output")
        shutil.move(produced, dst)


def _via_pandoc(src: str, dst: str) -> None:
    _require("pandoc")
    _run(["pandoc", src, "-o", dst])


# target extension → how to produce it, per source group
def _converter(src_ext: str, target_ext: str):
    if target_ext == ".png":
        return _to_png
    if src_ext in (".djvu", ".djv"):
        return _to_pdf_via_ddjvu
    if src_ext in (".epub", ".txt", ".md"):
        return _via_pandoc
    return _via_soffice


def convert_upload(path: str) -> str:
    """Convert ``path`` in place when needed; return the path to store.

    The source file is removed once the conversion succeeds, and kept when it
    fails so the upload can be retried after installing the missing tool.
    """
    src_ext = os.path.splitext(path)[1].lower()
    target_ext = converted_extension(src_ext)
    if target_ext is None:
        return path

    dst = os.path.splitext(path)[0] + target_ext
    convert = _converter(src_ext, target_ext)
    try:
        convert(path, dst)
    except Exception as exc:  # converter exit, missing tool, pillow/pandoc internals
        raise RuntimeError(f"Could not convert {src_ext} upload: {exc}") from exc

    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError(f"Could not convert {src_ext} upload: the converter produced no output.")

    logger.info("Converted upload %s → %s", src_ext, target_ext)
    try:
        os.remove(path)
    except OSError:
        pass
    return dst
