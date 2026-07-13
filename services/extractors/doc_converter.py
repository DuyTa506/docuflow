"""
DOC → DOCX Converter.

Uses LibreOffice in headless mode to convert legacy .doc files to .docx
so they can be processed by DocxExtractor.

Delegates to utils.soffice.run_soffice() which handles AF_UNIX socket
restrictions in sandboxed / containerised Linux environments via an
LD_PRELOAD shim compiled on first use.  On Windows, soffice is called
directly without the shim.
"""

import os
import subprocess
import sys
import tempfile
from typing import Optional

from utils.soffice import run_soffice


def convert_doc_to_docx(
    doc_path: str,
    output_dir: Optional[str] = None,
) -> str:
    """
    Convert a .doc file to .docx using LibreOffice headless.

    Args:
        doc_path: Path to the source .doc file.
        output_dir: Directory where the converted .docx will be placed.
                    Defaults to a temporary directory (caller must clean up).

    Returns:
        Absolute path to the resulting .docx file.

    Raises:
        FileNotFoundError: If the source .doc file does not exist.
        RuntimeError: If LibreOffice is not found or conversion fails.
    """
    if not os.path.isfile(doc_path):
        raise FileNotFoundError(f"Source .doc file not found: {doc_path}")

    if output_dir is None:
        # On Windows, LibreOffice fails silently when --outdir contains non-ASCII
        # characters (e.g. usernames with accents/apostrophes).  Use C:\Temp instead.
        if sys.platform == "win32":
            base = os.environ.get("TEMP", "C:\\Temp")
            # Fallback to C:\Temp if TEMP itself contains non-ASCII
            try:
                base.encode("ascii")
            except UnicodeEncodeError:
                base = "C:\\Temp"
            os.makedirs(base, exist_ok=True)
            output_dir = tempfile.mkdtemp(prefix="doc_convert_", dir=base)
        else:
            output_dir = tempfile.mkdtemp(prefix="doc_convert_")

    os.makedirs(output_dir, exist_ok=True)

    try:
        result = run_soffice(
            ["--headless", "--convert-to", "docx", "--outdir", output_dir, doc_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "LibreOffice (soffice) not found. "
            "Install LibreOffice and ensure 'soffice' is on your PATH."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("LibreOffice conversion timed out after 120 seconds.")

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"LibreOffice conversion failed (exit {result.returncode}): {stderr}")

    # Resolve output path
    base_name = os.path.splitext(os.path.basename(doc_path))[0]
    docx_path = os.path.join(output_dir, base_name + ".docx")

    if not os.path.isfile(docx_path):
        candidates = [f for f in os.listdir(output_dir) if f.lower().endswith(".docx")]
        if candidates:
            docx_path = os.path.join(output_dir, candidates[0])
        else:
            raise RuntimeError(
                f"LibreOffice ran successfully but no .docx file found in {output_dir}."
            )

    return docx_path
