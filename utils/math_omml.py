"""Convert LaTeX math snippets to OMML for embedding in Word documents."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_LATEX_INLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
_LATEX_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_OMML_NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
_DISPLAY_BRACKET = re.compile(r"^\\\[(.*)\\\]$", re.DOTALL)
_INLINE_PAREN = re.compile(r"^\\\((.*)\\\)$", re.DOTALL)


def normalize_latex_delimiters(text: str) -> str:
    """Strip OCR-style \\[...\\] / \\(...\\) wrappers so pandoc can convert."""
    t = (text or "").strip()
    if not t:
        return ""
    m = _DISPLAY_BRACKET.match(t)
    if m:
        return m.group(1).strip()
    m = _INLINE_PAREN.match(t)
    if m:
        return m.group(1).strip()
    return t


def is_pandoc_available() -> bool:
    return shutil.which("pandoc") is not None


def latex_to_omml_fragment(latex: str, *, display: bool = False) -> bytes | None:
    """
    Convert a LaTeX string to OMML XML bytes via pandoc.

    Returns None when pandoc is unavailable or conversion fails.
    """
    if not latex or not is_pandoc_available():
        return None
    body = normalize_latex_delimiters(latex)
    if not body:
        return None
    if display and not body.startswith("$$"):
        body = f"$$\n{body}\n$$"
    elif not display and not body.startswith("$"):
        body = f"${body}$"

    with tempfile.TemporaryDirectory() as tmpdir:
        md_file = Path(tmpdir) / "eq.md"
        docx_file = Path(tmpdir) / "eq.docx"
        md_file.write_text(body, encoding="utf-8")
        try:
            subprocess.run(
                ["pandoc", str(md_file), "-o", str(docx_file)],
                capture_output=True,
                check=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None

        try:
            from zipfile import ZipFile

            with ZipFile(docx_file) as zf:
                xml = zf.read("word/document.xml").decode("utf-8")
        except Exception:
            return None

    m = re.search(r"<m:oMathPara[^>]*>.*?</m:oMathPara>", xml, re.DOTALL)
    if not m:
        m = re.search(r"<m:oMath[^>]*>.*?</m:oMath>", xml, re.DOTALL)
    return m.group(0).encode("utf-8") if m else None


def omml_fragment_for_docx(omml_bytes: bytes) -> str:
    """Add Word math namespace so python-docx parse_xml accepts pandoc OMML."""
    omml_str = omml_bytes.decode("utf-8")
    if "xmlns:m=" in omml_str:
        return omml_str
    for tag in ("m:oMathPara", "m:oMath"):
        needle = f"<{tag}"
        if omml_str.startswith(needle):
            return omml_str.replace(needle, f"<{tag} {_OMML_NS}", 1)
    return f"<m:oMath {_OMML_NS}>{omml_str}</m:oMath>"


def wrap_as_equation_markdown(text: str) -> str:
    """Ensure equation text is wrapped in display LaTeX delimiters."""
    t = normalize_latex_delimiters(text)
    if not t:
        return ""
    if t.startswith("$$") and t.endswith("$$"):
        return t
    if t.startswith("$") and t.endswith("$"):
        return t
    return f"$${t}$$"
