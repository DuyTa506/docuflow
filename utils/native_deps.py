"""Startup check for native binaries the app shells out to.

Both are resolved with ``shutil.which`` at the moment they're used and degrade
*silently* when absent — ``markdown_pandoc``/``math_omml`` just report
``pandoc`` unavailable and fall back, and a missing LibreOffice only surfaces
when somebody uploads a .doc. Nothing declares them, so an incomplete
container image or a host whose conda PATH changed loses a feature with no
signal at all.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativeDep:
    name: str
    impact: str

    def probe(self) -> str:
        """Executable to look for. LibreOffice is configurable, so probe the
        configured path rather than the literal name."""
        if self.name == "soffice":
            from config.settings import settings

            return settings.libreoffice_path or "soffice"
        return self.name


NATIVE_DEPS: tuple[NativeDep, ...] = (
    NativeDep(
        name="pandoc",
        impact=(
            "DOCX export silently falls back to the python engine and loses "
            "OMML math formulas (utils/markdown_pandoc.py, utils/math_omml.py)"
        ),
    ),
    NativeDep(
        name="soffice",
        impact=(
            "DOC/DOCX conversion fails — uploads of those formats cannot be "
            "extracted (utils/soffice.py, set LIBREOFFICE_PATH to override)"
        ),
    ),
)


def missing_native_dependencies() -> list[NativeDep]:
    return [dep for dep in NATIVE_DEPS if shutil.which(dep.probe()) is None]


def log_native_dependency_warnings() -> list[str]:
    """Warn about each missing binary. Returns the messages emitted."""
    messages = []
    for dep in missing_native_dependencies():
        message = f"{dep.name} not found on PATH — {dep.impact}"
        logger.warning(message)
        messages.append(message)
    return messages
