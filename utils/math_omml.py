"""Convert LaTeX math snippets to OMML for embedding in Word documents."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_LATEX_INLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
_LATEX_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)


def is_pandoc_available() -> bool:
    return shutil.which("pandoc") is not None


def latex_to_omml_fragment(latex: str, *, display: bool = False) -> bytes | None:
    """
    Convert a LaTeX string to OMML XML bytes via pandoc.

    Returns None when pandoc is unavailable or conversion fails.
    """
    if not latex or not is_pandoc_available():
        return None
    body = latex.strip()
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

    m = re.search(r"<m:oMath[^>]*>.*?</m:oMath>", xml, re.DOTALL)
    if not m:
        m = re.search(r"<m:oMathPara[^>]*>.*?</m:oMathPara>", xml, re.DOTALL)
    return m.group(0).encode("utf-8") if m else None


def wrap_as_equation_markdown(text: str) -> str:
    """Ensure equation text is wrapped in display LaTeX delimiters."""
    t = (text or "").strip()
    if not t:
        return ""
    if t.startswith("$$") and t.endswith("$$"):
        return t
    if t.startswith("$") and t.endswith("$"):
        return t
    return f"$${t}$$"
