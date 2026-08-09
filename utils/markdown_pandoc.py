"""Markdown → DOCX via Pandoc (LaTeX → OMML equations)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from utils.markdown_docx import build_docx_bytes_from_markdown

logger = logging.getLogger(__name__)


def is_pandoc_available() -> bool:
    return shutil.which("pandoc") is not None


def markdown_to_docx_bytes(
    markdown: str,
    *,
    title: str | None = None,
    headings: list[str] | None = None,
) -> bytes:
    """
    Convert markdown to DOCX. Uses pandoc when available (preserves LaTeX as OMML),
    otherwise falls back to the python-docx renderer.
    """
    from config.settings import settings

    engine = settings.docx_export_engine
    if engine == "python":
        return build_docx_bytes_from_markdown(markdown, title=title, headings=headings)

    use_pandoc = engine == "pandoc" or (engine == "auto" and is_pandoc_available())
    if (
        use_pandoc
        and is_pandoc_available()
        and len(markdown or "") <= settings.export_pandoc_max_chars
    ):
        try:
            return _pandoc_convert(markdown, title=title, headings=headings)
        except Exception as exc:
            logger.warning("Pandoc conversion failed, falling back to python-docx: %s", exc)

    return build_docx_bytes_from_markdown(markdown, title=title, headings=headings)


def _pandoc_convert(
    markdown: str,
    *,
    title: str | None = None,
    headings: list[str] | None = None,
) -> bytes:
    prefix_parts = []
    if title:
        prefix_parts.append(f"# {title}\n")
    for h in headings or []:
        prefix_parts.append(f"## {h}\n")
    full_md = "".join(prefix_parts) + (markdown or "")

    with tempfile.TemporaryDirectory() as tmpdir:
        md_file = Path(tmpdir) / "input.md"
        docx_file = Path(tmpdir) / "output.docx"
        md_file.write_text(full_md, encoding="utf-8-sig")
        subprocess.run(
            ["pandoc", str(md_file), "-o", str(docx_file)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return docx_file.read_bytes()
