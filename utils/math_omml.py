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


_LATEX_COMMAND_UNICODE = {
    r"\times": "\u00d7",
    r"\cdot": "\u00b7",
    r"\pm": "\u00b1",
    r"\mp": "\u2213",
    r"\leq": "\u2264",
    r"\geq": "\u2265",
    r"\neq": "\u2260",
    r"\approx": "\u2248",
    r"\infty": "\u221e",
    r"\rightarrow": "\u2192",
    r"\to": "\u2192",
    r"\leftarrow": "\u2190",
    r"\Rightarrow": "\u21d2",
    r"\degree": "\u00b0",
    r"\circ": "\u00b0",
    r"\sum": "\u2211",
    r"\prod": "\u220f",
    r"\int": "\u222b",
    r"\sqrt": "\u221a",
    r"\alpha": "\u03b1",
    r"\beta": "\u03b2",
    r"\gamma": "\u03b3",
    r"\delta": "\u03b4",
    r"\epsilon": "\u03b5",
    r"\theta": "\u03b8",
    r"\lambda": "\u03bb",
    r"\mu": "\u00b5",
    r"\pi": "\u03c0",
    r"\rho": "\u03c1",
    r"\sigma": "\u03c3",
    r"\tau": "\u03c4",
    r"\phi": "\u03c6",
    r"\omega": "\u03c9",
    r"\Omega": "\u03a9",
    r"\Delta": "\u0394",
    r"\%": "%",
    r"\&": "&",
    r"\#": "#",
    r"\$": "$",
    r"\,": " ",
}
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")


def sanitize_latex_fragment(text: str) -> str:
    """Clean OCR LaTeX noise: collapse doubled braces (``{{x}}`` -> ``{x}``)."""
    t = (text or "").strip()
    prev = None
    while prev != t and ("{{" in t or "}}" in t):
        prev = t
        t = t.replace("{{", "{").replace("}}", "}")
    return t.strip()


# A bare variable — `$n$`, `$f$`, `$dx$`. Currency always starts with a digit,
# so a short letter-led identifier cannot be the `$5` case this guard exists for.
_MATH_VARIABLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,2}$")


def looks_like_math(text: str) -> bool:
    """Heuristic to avoid treating currency like ``$5`` as a math fragment.

    Accepts anything carrying LaTeX syntax, plus a bare short variable: a digest
    of a computer-architecture book says "a program of $n$ lines", and printing
    the dollars around it is worse than italicising the letter. Measured on the
    DOC_065 digest: 10 of the 10 surviving literal `$` were `$n$`, `$f$`, `$e$`.
    """
    if not text:
        return False
    if any(ch in text for ch in "\\^_{}") or _LATEX_CMD_RE.search(text):
        return True
    return bool(_MATH_VARIABLE_RE.match(text.strip()))


def has_latex_command(text: str) -> bool:
    """True when the fragment contains real LaTeX commands (needs OMML)."""
    return bool(_LATEX_CMD_RE.search(text or ""))


def replace_latex_commands(text: str) -> str:
    """Map common LaTeX commands to Unicode for the plain-text fallback."""
    if not text:
        return ""
    out = text
    for cmd, uni in _LATEX_COMMAND_UNICODE.items():
        out = out.replace(cmd, uni)
    out = out.replace("\\left", "").replace("\\right", "")
    return out


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
