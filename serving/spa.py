"""Serve the built Angular frontend from the API's own origin.

One origin is what lets `assets/env.json` hold the relative `/api/v2/` instead
of an absolute URL. An absolute one can only be right for a single caller:
`http://localhost:8022/api/v2/` serves an SSH port-forward and fails for every
colleague on the LAN, whose browser resolves `localhost` to their own machine.
A relative URL resolves against whatever host the browser used, so the same
build works through a LAN IP, a port-forward, and a later hostname or reverse
proxy without a rebuild or a per-machine config file.
"""

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

# Prefixes that belong to the API and must never fall through to the SPA.
# `/api` is the contract; the others are FastAPI's own and would otherwise be
# swallowed by the catch-all, because it is registered after them but matches
# anything they miss (a mistyped endpoint under a real prefix).
_API_PREFIXES = ("api/", "docs", "redoc", "openapi.json", "health")


def default_dist_dir() -> Path:
    """Where the build lands, overridable for a deployment that moves it."""
    override = os.getenv("DOCUFLOW_FRONTEND_DIST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "Fe-Library" / "dist"


def _resolve_within(root: Path, relative: str) -> Path | None:
    """Resolve `relative` under `root`, or None if it escapes or is missing.

    Resolving before comparing is the point: `assets/../../secret` only shows
    itself as an escape once the `..` segments are collapsed, and symlinks
    inside the build could otherwise point anywhere on disk.
    """
    try:
        candidate = (root / relative).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_file():
        return None
    if candidate != root and root not in candidate.parents:
        return None
    return candidate


def mount_spa(app: FastAPI, dist_dir: Path | None = None) -> bool:
    """Serve `dist_dir` for every path the API itself does not handle.

    Returns whether anything was mounted. A missing build is normal — the API
    is developed and deployed without one — so it logs and carries on rather
    than failing to start.
    """
    dist_dir = Path(dist_dir) if dist_dir is not None else default_dist_dir()
    try:
        root = dist_dir.resolve()
    except (OSError, ValueError):
        root = dist_dir
    index = root / "index.html"

    if not index.is_file():
        logger.info(
            "No frontend build at %s — API only. Build it with `ng build` or set "
            "DOCUFLOW_FRONTEND_DIST.",
            dist_dir,
        )
        return False

    # Registered last and matching everything, so every real route added before
    # it still wins. Angular does its own routing, hence a real page (not a
    # 404) for any path that is not a file in the build.
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith(_API_PREFIXES):
            raise HTTPException(status_code=404, detail="Not found")

        file = _resolve_within(root, full_path)
        if file is not None:
            return FileResponse(file)

        return FileResponse(index)

    logger.info("Serving frontend from %s", root)
    return True
